"""Train ACT / PA-ACT on one LIBERO-Spatial task.

Two stages, both on the 50 Standard demos only:
  1. selection: train on 45 demos for `epochs`, keep the epoch with the
     lowest loss on the 5 held-out demos
  2. refit: retrain from scratch on all 50 demos for that many epochs

Example:
  python train.py --config configs/pa_act.json \
      --task pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
      --data data/libero_spatial --out checkpoints/pa_act/table_center
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from policy import ACTPolicy
from pa_act.data import LiberoChunkDataset, inspect_demonstrations, normalization_stats
from pa_act.optim import AccumulatingOptimizer, save_training_state, restore_training_state


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def run_epoch(policy, loader, opt, accumulation, amp_dtype, use_amp, device):
    policy.train()
    total, count = 0., 0
    for step, batch in enumerate(loader):
        q, images, actions, pad = [x.to(device, non_blocking=True) for x in batch]
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            loss = policy(q, images, actions, pad)['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('loss is NaN/inf')
        opt.backward(loss, len(q))
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            opt.step()
        total += float(loss.detach()) * len(q)
        count += len(q)
    return total / count


@torch.no_grad()
def validate(policy, loader, amp_dtype, use_amp, device):
    policy.eval()
    total, count = 0., 0
    for batch in loader:
        q, images, actions, pad = [x.to(device, non_blocking=True) for x in batch]
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            loss = policy(q, images, actions, pad)['loss']
        total += float(loss) * len(q)
        count += len(q)
    return total / count


def train(cfg, train_cfg, hdf5, out, device='cuda', resume=False):
    policy_cfg = cfg['policy']
    out.mkdir(parents=True, exist_ok=True)
    seed = train_cfg['seed']
    amp_dtype = torch.bfloat16 if train_cfg['amp_dtype'] == 'bfloat16' else torch.float16
    use_amp = train_cfg['amp']
    accumulation = train_cfg['gradient_accumulation']
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    keys = inspect_demonstrations(hdf5, policy_cfg, required=train_cfg['demonstrations'])
    order = np.random.default_rng(seed).permutation(len(keys))
    n_val = train_cfg['validation_demos']
    val_keys = [keys[i] for i in order[:n_val]]
    train_keys = [keys[i] for i in order[n_val:]]
    (out / 'policy.json').write_text(json.dumps(policy_cfg, indent=2))

    latest = out / 'latest.pt'
    state = None
    if resume and latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)['metadata']
    best_epoch, best_loss = (state['best_epoch'], state['best_loss']) if state else (0, float('inf'))
    t0 = time.time()

    with (out / 'training.jsonl').open('a' if resume else 'w') as log:
        for stage in ('selection', 'refit'):
            if state and state['stage'] == 'refit' and stage == 'selection':
                continue
            seed_everything(seed)
            demos = train_keys if stage == 'selection' else keys
            stats = normalization_stats(hdf5, demos, policy_cfg, train_cfg['std_floor'])
            gen = torch.Generator().manual_seed(seed)
            loader = DataLoader(LiberoChunkDataset(hdf5, demos, policy_cfg, stats),
                                batch_size=train_cfg['batch_size'], shuffle=True,
                                num_workers=train_cfg['num_workers'], pin_memory=True,
                                worker_init_fn=seed_worker, generator=gen)
            val_loader = DataLoader(LiberoChunkDataset(hdf5, val_keys, policy_cfg, stats),
                                    batch_size=train_cfg['validation_batch_size'],
                                    num_workers=train_cfg['num_workers'], pin_memory=True)

            policy = ACTPolicy({**policy_cfg, 'device': device,
                                'samplewise_nll': train_cfg['samplewise_nll']})
            if policy_cfg.get('phase_aware_wrist_attention'):
                policy.set_observation_normalization(stats['qpos_mean'], stats['qpos_std'])
            optimizer = policy.configure_optimizers()
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
            opt = AccumulatingOptimizer(policy, optimizer, scaler)

            epochs = train_cfg['epochs'] if stage == 'selection' else best_epoch + 1
            start = 0
            if state and state['stage'] == stage:
                restore_training_state(latest, policy, optimizer, scaler, gen)
                start = state['epoch'] + 1

            for epoch in range(start, epochs):
                train_loss = run_epoch(policy, loader, opt, accumulation, amp_dtype, use_amp, device)
                val_loss = None
                if stage == 'selection':
                    val_loss = validate(policy, val_loader, amp_dtype, use_amp, device)
                    if val_loss < best_loss:
                        best_loss, best_epoch = val_loss, epoch
                row = dict(stage=stage, epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                           minutes=(time.time() - t0) / 60)
                save_training_state(latest, policy, optimizer, scaler, gen,
                                    dict(stage=stage, epoch=epoch, best_epoch=best_epoch, best_loss=best_loss))
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(row, flush=True)

            if stage == 'refit':
                weights = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                torch.save(weights, out / 'policy_best.ckpt')
                np.savez(out / 'stats.npz', **stats)
                n_params = sum(p.numel() for p in policy.parameters())
            del policy, optimizer, scaler, opt, loader, val_loader
            torch.cuda.empty_cache()

    summary = dict(selected_epochs=best_epoch + 1, best_val_loss=best_loss, parameters=n_params,
                   train_demos=train_keys, val_demos=val_keys, minutes=(time.time() - t0) / 60)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--task', required=True)
    ap.add_argument('--data', default='data/libero_spatial', type=Path,
                    help='folder with the LIBERO-Spatial *_demo.hdf5 files')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--train-config', default='configs/train.json', type=Path)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    train_cfg = json.loads(args.train_config.read_text())
    hdf5 = args.data / f'{args.task}_demo.hdf5'
    print(train(cfg, train_cfg, hdf5, args.out, args.device, args.resume))
