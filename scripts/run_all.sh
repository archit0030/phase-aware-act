#!/usr/bin/env bash
# Full pipeline: train 3 methods x 2 tasks, then evaluate on Standard / P2 / P1.
#   bash scripts/run_all.sh            # train + evaluate
#   bash scripts/run_all.sh --eval     # only evaluate (uses checkpoints/)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

METHODS="act_k50 act_k10 pa_act"
declare -A TASKS=(
    [table_center]=pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
    [between_plate_ramekin]=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
)

if [ "${1:-}" != "--eval" ]; then
    for m in $METHODS; do
        for short in "${!TASKS[@]}"; do
            python train.py --config configs/$m.json --task "${TASKS[$short]}" \
                --data data/libero_spatial --out checkpoints/$m/$short
        done
    done
fi

for split in Standard P2 P1; do
    for m in $METHODS; do
        python evaluate.py --method $m --split $split --episodes 20 --out runs/$m/$split
    done
done
python scripts/summarize.py --root runs --out runs/summary
