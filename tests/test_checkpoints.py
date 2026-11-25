import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [(m, t) for m in ('act_k50', 'act_k10', 'pa_act') for t in ('between_plate_ramekin', 'table_center')]


@pytest.mark.parametrize('method,task', CASES)
def test_checkpoint_loads(method, task):
    ckpt = ROOT / 'checkpoints' / method / task
    if not (ckpt / 'policy_best.ckpt').exists():
        pytest.skip('checkpoints not downloaded')
    from evaluate import load_policy
    cfg = json.loads((ROOT / 'configs' / f'{method}.json').read_text())
    policy, adapter = load_policy(cfg, ckpt, 'cpu')   # strict=True inside
    assert adapter.stats['action_mean'].shape == (7,)
