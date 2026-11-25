# AR523 Project: Phase-Aware ACT (PA-ACT) on LIBERO-PRO

This project tests whether Action Chunking with Transformers (ACT) still works when objects in the scene are moved. It compares three policies on two LIBERO-Spatial pick-and-place tasks:

| Method | What it is |
|---|---|
| `act_k50` | Plain ACT, predicts 50-step action chunks (the standard setting) |
| `act_k10` | Plain ACT with 10-step chunks (shorter horizon, re-plans more often) |
| `pa_act` | **PA-ACT**: ACT + Laplace uncertainty head + wrist-camera attention bias + DUTE ensembling |

All three are trained only on the original (Standard) layouts. They are then tested on:

- **Standard**: the same layouts as in training.
- **P2** (LIBERO-PRO "swap"): the plate and the cookie box swap places. The instruction is the same, but the plate is somewhere new.
- **P1** (LIBERO-PRO "task"): a different bowl has to be picked.

Tasks (50 demos each, from LIBERO-Spatial):

- `table_center`: *pick up the black bowl from table center and place it on the plate*
- `between_plate_ramekin`: *pick up the black bowl between the plate and the ramekin and place it on the plate*

## How PA-ACT differs from ACT

1. **Uncertainty head.** The action head outputs a mean and a Laplace scale *b* for every action dimension, trained with the Laplace NLL `|a - mu| / b + log b` instead of L1. This adds 1,799 parameters (256×7 + 7) to ACT's 41.78 M.
2. **Wrist attention bias.** For the first 8 of the 50 action queries, `log(0.1)` is added to the decoder's attention scores on every agentview image token, so the next few actions rely mostly on the wrist camera. The other 42 queries see both cameras normally. See `detr/models/transformer.py` (`wrist_attention_bias`).
3. **DUTE ensembling.** Overlapping chunks are mixed with weights `softmax(-m·age - λ1·b - λ2·||a_i - a_fresh||²)`, so old chunks that are uncertain or that disagree with the newest prediction count less. Plain ACT uses only `exp(-m·age)`. See `pa_act/temporal.py`.

`configs/pa_act_phase.json` is a variant that switches the wrist bias off once the gripper closes (finger width < 60 mm). It is implemented and unit-tested but **was not trained or evaluated**.

## Results

20 episodes per task, 2 tasks, so 40 episodes per cell. Seed 0, LIBERO initial states 0–19.

| Method | Standard | P2 (plate moved) | P1 (other bowl) |
|---|---|---|---|
| ACT k=50 | 38/40 (95.0%) | 0/40 | 0/40 |
| ACT k=10 | 38/40 (95.0%) | 1/40 (2.5%) | 0/40 |
| PA-ACT   | 37/40 (92.5%) | 0/40 | 0/40 |

What we found:

- All three methods do well on the training layouts and almost always fail when the plate moves.
- On P2, every method **reached and touched the bowl in 40/40 episodes** (`results/summary/p2_reach_vs_place.csv`). The failure happens after the grasp, at placement.
- PA-ACT does shift attention to the wrist camera as intended: 79.5% / 85.1% of short-horizon attention on the wrist, against 57.7% / 48.5% for ACT k=50 (`results/summary/wrist_attention.csv`).
- In the episode-0 replays, PA-ACT carried the bowl to where the plate *used to be*: 28 mm from the old spot and 149 mm from the moved plate. After the grasp the bowl blocks the wrist camera, and the bias still pushes the short-horizon queries away from agentview, the only camera that can see the new plate. This is what motivated the gripper-phase variant above.

Limitations: 2 closely related tasks, one training seed, 20 initial states per task. Treat this as a pilot study, not a benchmark score.

## Repository layout

```
configs/            one JSON per method (policy + ensembling), train.json for training settings
detr/               ACT network (CVAE + transformer). *_standard.py, backbone.py and
                    position_encoding.py are unchanged upstream ACT; detr_vae.py and
                    transformer.py add the PA-ACT changes
pa_act/             data loading, temporal ensembling / DUTE, attention probes, helpers
policy.py           policy wrapper: normalisation, L1 / Laplace loss, inference
train.py            train one method on one task
evaluate.py         run evaluation episodes on Standard / P1 / P2
assets/             BDDL + initial-state files for the 2 tasks in each split
scripts/
  run_all.sh        train everything, then evaluate everything
  summarize.py      build the success-rate tables
  replay_episode.py replay a logged episode (trace, plot and optional video)
tests/              unit tests (model, ensembling, data loader, checkpoints)
results/
  summary/          success rates (with Wilson 95% CI), per-task rates, attention, reach
  <method>/training/<task>/   training log + selected epoch
  <method>/eval/<split>/      summary.json, episodes.jsonl, executed actions of episode 0
  episode_replays/  episode-0 traces and trajectory plots (bowl, gripper, plate)
  videos/<method>/  success and failure clips (+ poster frame)
```

## Setup

Tested on Ubuntu 22.04, Python 3.10, PyTorch 2.5.1 + CUDA 12.4, on one RTX 4060 Ti (8 GB).

```bash
bash setup.sh               # venv + PyTorch + LIBERO-PRO
bash setup.sh --with-data   # also downloads the LIBERO-Spatial demos (needed only for training)
source env.sh
```

## Downloads

**Checkpoints.** The 6 trained models (3 methods × 2 tasks, about 160 MB each) are too big for git, so they are attached to the [v1.0 release](https://github.com/archit0030/phase-aware-act/releases/tag/v1.0). Each `.tar` holds `policy_best.ckpt`, `stats.npz` and `policy.json`:

```bash
gh release download v1.0 -R archit0030/phase-aware-act -p '*.tar' -D checkpoints
for f in checkpoints/*.tar; do tar -xf "$f" -C checkpoints && rm "$f"; done
# -> checkpoints/act_k50/table_center/policy_best.ckpt, checkpoints/pa_act/..., ...
```

**Results and videos.** The same results as in `results/`, plus the videos, are also in Google Drive: [AR523_PA-ACT_Results](https://drive.google.com/drive/folders/1h3XUWm2n_kxjkLnerlORcKpfJPlSmVzR) (`results_tables_logs_replays.zip` and `videos/`).

## Evaluate

```bash
source env.sh
python -m pytest                                          # unit tests (~10 s)

# one method on one split: 20 episodes per task, writes runs/<method>/<split>/
python evaluate.py --method pa_act --split P2 --episodes 20

# quick check, 1 episode
python evaluate.py --method act_k50 --split Standard --tasks table_center --episodes 1

# all methods x all splits, then the results table
bash scripts/run_all.sh --eval
```

Each run writes `summary.json` (success rate) and `episodes.jsonl` (per episode: success, steps, whether the bowl was touched, closest distance to the bowl, final bowl-to-plate distance). Add `--log-steps` to also save per-step actions and attention.

Running `evaluate.py` with the checkpoints above gives exactly the same episodes as the saved results. For example, episode 0 of `table_center` ends at step 98 / 91 / 102 (Standard) and 220 (P2, failure) for ACT k=50 / ACT k=10 / PA-ACT.

## Train

```bash
python train.py --config configs/pa_act.json \
    --task pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
    --data data/libero_spatial --out checkpoints/pa_act/table_center
```

Training follows two stages:

1. Train on 45 demos and pick the epoch with the lowest loss on the other 5 (10 epochs maximum).
2. Retrain on all 50 demos for that many epochs.

It uses AdamW (lr 1e-5), batch 8, bf16 and KL weight 10. A full run (10 selection epochs + refit) took about 1.5 h per model on the 4060 Ti. `bash scripts/run_all.sh` trains and evaluates all 6 models.

## Videos

All clips are replays of the logged episode 0, rendered from the recorded actions with `scripts/replay_episode.py --video`:

| Method | Success | Failure |
|---|---|---|
| ACT k=50 | `success_standard_table_center.mp4` | `failure_p2_table_center.mp4` (released 61 mm short of the moved plate), `failure_p2_between_plate_ramekin.mp4` (bowl taken to the old plate spot, 161 mm from the moved plate) |
| ACT k=10 | `success_standard_table_center.mp4` | `failure_p2_table_center.mp4` |
| PA-ACT | `success_standard_table_center.mp4` | `failure_p2_table_center.mp4` (bowl taken to the old plate spot) |

## Credits

- ACT: Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023, https://github.com/tonyzhaozh/act. The `detr/` code is based on it (license in `detr/LICENSE`).
- LIBERO: Liu et al., 2023, https://github.com/Lifelong-Robot-Learning/LIBERO.
- LIBERO-PRO: https://github.com/Zxy-MLlab/LIBERO-PRO. The P1/P2 task files in `assets/` come from it (CC-BY-4.0).

## License

Released under the [MIT License](LICENSE). The code in `detr/` keeps its original
notices (ACT, MIT, see `detr/LICENSE`; some files are derived from Facebook's DETR).
The LIBERO-PRO task files in `assets/` are under CC BY 4.0.
