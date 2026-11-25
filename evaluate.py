"""Evaluate a trained ACT / PA-ACT checkpoint on LIBERO-Spatial or LIBERO-PRO.

Splits (files in assets/<split>/):
  Standard  original LIBERO-Spatial layouts (same as training)
  P2        LIBERO-PRO "swap": plate and cookie box exchange places, same goal
  P1        LIBERO-PRO "task": a different bowl must be picked

Success is env.check_success(). Each episode: 10 settle steps with the gripper
open, then up to 220 policy steps. A new chunk is predicted at every step and
combined with the older chunks by temporal ensembling (static exp(-m*age) or DUTE).

Example:
  python evaluate.py --method pa_act --split P2 --episodes 20
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from policy import ACTPolicy
from pa_act.temporal import TemporalEnsembler
from pa_act.diagnostics import ContactDiagnostics, position_telemetry
from pa_act.math_utils import validate_camera_contract

ROOT = Path(__file__).resolve().parent
TASKS = {
    'between_plate_ramekin': 'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate',
    'table_center': 'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate',
}
SETTLE_STEPS, MAX_STEPS = 10, 220
OPEN_GRIPPER = np.array([0., 0., 0., 0., 0., 0., -1.])


def read_states(path):
    # LIBERO .pruned_init files are pickled numpy arrays
    states = torch.load(path, map_location='cpu', weights_only=False)
    return np.asarray(states.numpy() if isinstance(states, torch.Tensor) else states)


def task_spec(short_name, split):
    name = TASKS[short_name]
    return dict(
        id=name, short=short_name, split=split,
        bddl=ROOT / 'assets' / split / f'{name}.bddl',
        init_states=ROOT / 'assets' / split / f'{name}.pruned_init',
        # in P1 the instruction asks for the second bowl
        target_position_key='akita_black_bowl_2_pos' if split == 'P1' else 'akita_black_bowl_1_pos',
        placement_position_key='plate_1_pos')


class ObservationAdapter:
    """Turns a LIBERO observation into normalised (qpos, images) tensors."""

    def __init__(self, cfg, stats_path, device):
        validate_camera_contract(cfg)
        self.cfg, self.device = cfg, device
        with np.load(stats_path) as s:
            self.stats = {k: torch.as_tensor(s[k], device=device, dtype=torch.float32)
                          for k in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std')}

    def encode(self, obs):
        qpos = np.concatenate([np.asarray(obs[k]).reshape(-1) for k in self.cfg['qpos_obs_keys']])
        qpos = torch.as_tensor(qpos, device=self.device, dtype=torch.float32)
        qpos = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        images = []
        for key in self.cfg['image_obs_keys']:
            rgb = np.asarray(obs[key])
            if self.cfg.get('flip_images_vertical', True):
                rgb = rgb[::-1]
            images.append(torch.as_tensor(rgb.copy(), device=self.device).permute(2, 0, 1))
        images = torch.stack(images).float().div(255.)
        return qpos[None], images[None]

    def decode_action(self, action):
        return action * self.stats['action_std'] + self.stats['action_mean']


def contact_geoms(env, target_key):
    obj = target_key[:-len('_pos')]
    return dict(gripper_geom_names=list(env.robots[0].gripper.contact_geoms),
                object_geom_names=list(env.env.objects_dict[obj].contact_geoms))


def load_policy(cfg, ckpt_dir, device):
    policy = ACTPolicy({**cfg['policy'], 'device': str(device)})
    state = torch.load(ckpt_dir / 'policy_best.ckpt', map_location='cpu', weights_only=True)
    policy.load_state_dict(state, strict=True)
    policy.to(device).eval()
    adapter = ObservationAdapter(cfg['policy'], ckpt_dir / 'stats.npz', device)
    if cfg['policy'].get('phase_aware_wrist_attention'):
        policy.set_observation_normalization(adapter.stats['qpos_mean'], adapter.stats['qpos_std'])
    return policy, adapter


@torch.inference_mode()
def run_episode(env, policy, adapter, buffer, task, init_state, seed, logger=None, step_log=None):
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    buffer.reset()
    contact = ContactDiagnostics(env, task)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(OPEN_GRIPPER)
    start = position_telemetry(obs, task)

    success = bool(env.check_success())
    steps, latencies, min_dist = 0, [], float('inf')
    for t in range(MAX_STEPS):
        if success:
            break
        qpos, images = adapter.encode(obs)
        tic = time.perf_counter()
        pred = policy(qpos, images, return_uncertainty=True)
        result = buffer.update(t, pred.mean[0], None if pred.scale is None else pred.scale[0])
        latencies.append(time.perf_counter() - tic)

        eef = np.asarray(obs['robot0_eef_pos'])
        bowl = np.asarray(obs[task['target_position_key']])
        dist_mm = float(np.linalg.norm(eef - bowl) * 1000)
        min_dist = min(min_dist, dist_mm)

        action = adapter.decode_action(result.action).cpu().numpy()
        obs, _, done, _ = env.step(np.clip(action, -1., 1.))
        touched = contact.observe(obs)
        steps += 1
        success = bool(env.check_success())

        if step_log is not None:
            row = dict(split=task['split'], task=task['id'], t=t, n_eff=float(result.effective_horizon),
                       eef_target_distance_mm=dist_mm, contact=touched['contact'],
                       gripper_open=(bool(policy.gripper_open_from_qpos(qpos)[0])
                                     if policy.model.phase_aware_wrist_attention else None))
            row.update(position_telemetry(obs, task))
            if logger is not None:
                row.update(logger.observe(t, pred, result))
            step_log.write(json.dumps(row) + '\n')
        if done:
            break

    final = position_telemetry(obs, task)
    bowl_xy, plate_xy = np.asarray(final['target_object_position_m'][:2]), np.asarray(final['placement_target_position_m'][:2])
    return dict(success=success, steps=steps, touched_bowl=contact.seen,
                min_eef_bowl_distance_mm=min_dist,
                final_bowl_plate_xy_mm=float(np.linalg.norm(bowl_xy - plate_xy) * 1000),
                initial_plate_position_m=start['placement_target_position_m'],
                policy_hz=len(latencies) / sum(latencies) if latencies else None)


def evaluate(method, split, tasks, episodes, ckpt_root, out, device, seed=0, log_steps=False):
    from libero.libero.envs import OffScreenRenderEnv
    cfg = json.loads((ROOT / 'configs' / f'{method}.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    totals = defaultdict(lambda: [0, 0])
    step_file = (out / 'timesteps.jsonl').open('w') if log_steps else None

    with (out / 'episodes.jsonl').open('w') as ep_file:
        for short in tasks:
            task = task_spec(short, split)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            policy, adapter = load_policy(cfg, ckpt_root / method / short, device)
            buffer = TemporalEnsembler(cfg['policy']['num_queries'], **cfg['ensemble'])
            logger = None
            if log_steps:
                from pa_act.probes import RolloutLogger
                logger = RolloutLogger(policy)
            states = read_states(task['init_states'])
            env = OffScreenRenderEnv(bddl_file_name=str(task['bddl']),
                                     camera_names=cfg['policy']['camera_names'],
                                     camera_heights=128, camera_widths=128,
                                     horizon=MAX_STEPS + SETTLE_STEPS + 1)
            try:
                env.seed(seed)
                env.reset()
                task['contact'] = contact_geoms(env, task['target_position_key'])
                for ep in range(episodes):
                    res = run_episode(env, policy, adapter, buffer, task, states[ep], seed + ep,
                                      logger, step_file)
                    totals[short][0] += res['success']
                    totals[short][1] += 1
                    row = dict(method=method, split=split, task=task['id'], episode=ep, **res)
                    ep_file.write(json.dumps(row) + '\n')
                    ep_file.flush()
                    print(f"{method} {split} {short} ep{ep:02d} success={res['success']} steps={res['steps']}", flush=True)
            finally:
                if logger is not None:
                    logger.close()
                env.close()
            del policy

    if step_file:
        step_file.close()
    wins = sum(v[0] for v in totals.values())
    n = sum(v[1] for v in totals.values())
    summary = dict(method=method, split=split, successes=wins, episodes=n,
                   success_rate_pct=100. * wins / n,
                   per_task={k: dict(successes=v[0], episodes=v[1]) for k, v in totals.items()})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, choices=['act_k50', 'act_k10', 'pa_act', 'pa_act_phase'])
    ap.add_argument('--split', required=True, choices=['Standard', 'P1', 'P2'])
    ap.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS))
    ap.add_argument('--episodes', type=int, default=20, help='initial states 0..N-1 (max 50)')
    ap.add_argument('--ckpt', default=ROOT / 'checkpoints', type=Path)
    ap.add_argument('--out', type=Path, help='default: runs/<method>/<split>')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-steps', action='store_true', help='also write per-step timesteps.jsonl')
    args = ap.parse_args()
    out = args.out or ROOT / 'runs' / args.method / args.split
    print(json.dumps(evaluate(args.method, args.split, args.tasks, args.episodes, args.ckpt,
                              out, args.device, args.seed, args.log_steps), indent=2))
