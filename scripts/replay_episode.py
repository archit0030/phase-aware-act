"""Replay a logged evaluation episode in the simulator.

Takes the executed actions saved in results/<method>/eval/<split>/episode0_actions.jsonl,
steps them through LIBERO again and writes
  trace.jsonl / trajectory.png / summary.json   (always)
  <name>.mp4 + <name>.png poster                (with --video)

The replay is checked against the logged end-effector-to-bowl distance, so it
reproduces the original rollout exactly (no new policy inference).

Example:
  python scripts/replay_episode.py --method pa_act --split P2 --task table_center --video
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import task_spec, read_states, SETTLE_STEPS, MAX_STEPS, OPEN_GRIPPER  # noqa: E402

SIZE, BAR, FPS = 448, 64, 20
LABELS = dict(act_k50='ACT (k=50)', act_k10='ACT (k=10)', pa_act='PA-ACT')
SPLIT_TEXT = dict(Standard='Standard layout',
                  P2='LIBERO-PRO P2: plate and cookie box swapped',
                  P1='LIBERO-PRO P1: different target bowl')


class Renderer:
    """Side-by-side agentview + wrist frames at a higher resolution than the policy sees."""

    def __init__(self, model):
        import mujoco
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, SIZE)
        model.vis.global_.offheight = max(model.vis.global_.offheight, SIZE)
        self.r = mujoco.Renderer(model, SIZE, SIZE)

    def frame(self, data, title, status, colour):
        from PIL import Image, ImageDraw, ImageFont
        views = []
        for cam in ('agentview', 'robot0_eye_in_hand'):
            self.r.update_scene(data, camera=cam)
            views.append(self.r.render().copy())
        img = np.concatenate(views, axis=1)
        h, w, _ = img.shape
        canvas = Image.new('RGB', (w, h + BAR), (24, 28, 36))
        canvas.paste(Image.fromarray(img), (0, BAR))
        draw = ImageDraw.Draw(canvas)
        try:
            big = ImageFont.truetype('DejaVuSans.ttf', 22)
            small = ImageFont.truetype('DejaVuSans.ttf', 16)
        except OSError:
            big = small = ImageFont.load_default()
        draw.text((14, 8), title, font=big, fill=(240, 240, 240))
        draw.text((14, 36), status, font=small, fill=colour)
        draw.text((10, BAR + 8), 'agentview camera', font=small, fill=(255, 255, 255))
        draw.text((w // 2 + 10, BAR + 8), 'wrist camera', font=small, fill=(255, 255, 255))
        return np.asarray(canvas)


def save_video(path, frames, poster_index):
    import imageio
    from PIL import Image
    frames = frames + [frames[-1]] * FPS  # hold the last frame for a second
    with imageio.get_writer(path, fps=FPS, codec='libx264', quality=8,
                            pixelformat='yuv420p', macro_block_size=16) as w:
        for f in frames:
            w.append_data(f)
    Image.fromarray(frames[min(poster_index, len(frames) - 1)]).save(path.with_suffix('.png'))


def plot(trace, plate, start_height, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eef = np.array([r['eef_m'] for r in trace]) * 1000
    bowl = np.array([r['bowl_m'] for r in trace]) * 1000
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(eef[:, 0], eef[:, 1], label='end effector')
    ax[0].plot(bowl[:, 0], bowl[:, 1], label='bowl')
    ax[0].scatter(plate[0] * 1000, plate[1] * 1000, marker='x', color='k', label='plate')
    ax[0].set(xlabel='x (mm)', ylabel='y (mm)')
    ax[0].legend(fontsize=8)
    ax[1].plot(bowl[:, 2] - start_height * 1000)
    ax[1].set(xlabel='step', ylabel='bowl lift (mm)')
    ax[2].plot([r['gripper_width_mm'] for r in trace])
    ax[2].set(xlabel='step', ylabel='gripper width (mm)')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def replay(method, split, short, out, video=False, seed=0):
    from libero.libero.envs import OffScreenRenderEnv
    task = task_spec(short, split)
    log = ROOT / 'results' / method / 'eval' / split / 'episode0_actions.jsonl'
    rows = [r for r in map(json.loads, log.read_text().splitlines()) if r['task'] == short]
    with np.load(ROOT / 'checkpoints' / method / short / 'stats.npz') as s:
        mean, std = s['action_mean'], s['action_std']
    states = read_states(task['init_states'])
    key = task['target_position_key']

    env = OffScreenRenderEnv(bddl_file_name=str(task['bddl']),
                             camera_names=['agentview', 'robot0_eye_in_hand'],
                             camera_heights=128, camera_widths=128,
                             horizon=MAX_STEPS + SETTLE_STEPS + 1)
    trace, errors, frames, success_step = [], [], [], None
    title = f'{LABELS[method]}  |  {SPLIT_TEXT[split]}'
    try:
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(OPEN_GRIPPER)
        start_height = float(obs[key][2])
        cam = Renderer(env.sim.model._model) if video else None
        for row in rows:
            eef, bowl = np.asarray(obs['robot0_eef_pos']), np.asarray(obs[key])
            errors.append(abs(np.linalg.norm(eef - bowl) * 1000 - row['eef_target_distance_mm']))
            if video:
                done = success_step is not None
                status = f"step {row['t']:3d}   " + ('SUCCESS: bowl on plate' if done else 'running')
                frames.append(cam.frame(env.sim.data._data, title, status,
                                        (120, 220, 140) if done else (200, 200, 200)))
            action = np.asarray(row['action'], dtype=np.float32) * std + mean
            obs, _, _, _ = env.step(np.clip(action, -1., 1.))
            ok = bool(env.check_success())
            if ok and success_step is None:
                success_step = row['t']
            trace.append(dict(t=row['t'], eef_m=obs['robot0_eef_pos'].tolist(), bowl_m=obs[key].tolist(),
                              plate_m=obs['plate_1_pos'].tolist(),
                              gripper_width_mm=float(np.abs(obs['robot0_gripper_qpos']).sum() * 1000),
                              success=ok))
        success = bool(env.check_success())
        last = trace[-1]
        dist = float(np.linalg.norm(np.asarray(last['bowl_m'][:2]) - np.asarray(last['plate_m'][:2])) * 1000)
        if video:
            end = 'SUCCESS: bowl on plate' if success else f'FAILED: bowl ends {dist:.0f} mm from the plate'
            frames.append(cam.frame(env.sim.data._data, title, f'step {len(rows):3d}   {end}',
                                    (120, 220, 140) if success else (255, 120, 110)))
    finally:
        env.close()

    out.mkdir(parents=True, exist_ok=True)
    with (out / 'trace.jsonl').open('w') as f:
        for r in trace:
            f.write(json.dumps(r) + '\n')
    summary = dict(method=method, split=split, task=short, episode=0, steps=len(trace),
                   success=success, success_step=success_step,
                   max_bowl_lift_mm=1000 * (max(r['bowl_m'][2] for r in trace) - start_height),
                   final_bowl_plate_xy_mm=dist,
                   max_replay_error_mm=float(max(errors)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot(trace, last['plate_m'], start_height, f'{split}: {LABELS[method]}, episode 0, success={success}',
         out / 'trajectory.png')
    if video:
        name = out / f'{method}_{split.lower()}_{short}.mp4'
        save_video(name, frames, int(len(frames) * 0.6) if success else len(frames) - 1)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', required=True, choices=list(LABELS))
    ap.add_argument('--split', required=True, choices=list(SPLIT_TEXT))
    ap.add_argument('--task', default='table_center', choices=['table_center', 'between_plate_ramekin'])
    ap.add_argument('--out', type=Path)
    ap.add_argument('--video', action='store_true')
    args = ap.parse_args()
    out = args.out or ROOT / 'runs' / 'replays' / args.task / args.split / args.method
    print(json.dumps(replay(args.method, args.split, args.task, out, args.video), indent=2))
