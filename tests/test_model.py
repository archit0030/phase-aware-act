import json
import math
from pathlib import Path

import pytest
import torch

from policy import ACTPolicy
from detr.models.transformer import wrist_attention_bias

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def make(method, **extra):
    cfg = json.loads((CONFIGS / f'{method}.json').read_text())['policy']
    torch.manual_seed(0)
    return ACTPolicy({**cfg, 'device': 'cpu', **extra})


def n_params(policy):
    return sum(p.numel() for p in policy.parameters())


def fake_batch(k, batch=2):
    qpos = torch.randn(batch, 9)
    images = torch.rand(batch, 2, 3, 128, 128)
    actions = torch.randn(batch, k, 7)
    pad = torch.zeros(batch, k, dtype=torch.bool)
    pad[:, -3:] = True
    return qpos, images, actions, pad


def test_parameter_counts():
    k50, k10, pa = make('act_k50'), make('act_k10'), make('pa_act')
    assert n_params(k50) == 41_782_920
    # 40 fewer learned queries of size 256
    assert n_params(k50) - n_params(k10) == 40 * 256
    # PA-ACT only widens the action head: 256*7 weights + 7 biases
    assert n_params(pa) - n_params(k50) == 256 * 7 + 7


@pytest.mark.parametrize('method,k', [('act_k50', 50), ('act_k10', 10), ('pa_act', 50)])
def test_forward_train_and_infer(method, k):
    policy = make(method)
    qpos, images, actions, pad = fake_batch(k)
    out = policy(qpos, images, actions, pad)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    policy.eval()
    with torch.no_grad():
        pred = policy(qpos, images, return_uncertainty=True)
    assert pred.mean.shape == (2, k, 7)
    if method == 'pa_act':
        assert pred.scale.shape == (2, k, 7)
        assert (pred.scale > 0).all()
    else:
        assert pred.scale is None


def test_uncertainty_head_starts_at_b_equal_one():
    policy = make('pa_act').eval()
    qpos, images, _, _ = fake_batch(50)
    with torch.no_grad():
        pred = policy(qpos, images, return_uncertainty=True)
    assert torch.allclose(pred.scale, torch.ones_like(pred.scale))


def test_wrist_bias_layout():
    # two cameras with 4x4 feature maps, wrist is camera 1
    bias = wrist_attention_bias(50, 8, [4, 4], 4, 1, 0.1, device='cpu', dtype=torch.float32)
    assert bias.shape == (50, 2 + 4 * 8)
    assert torch.all(bias[8:] == 0)                      # long-horizon queries untouched
    assert torch.all(bias[:8, :2] == 0)                  # latent + proprio tokens untouched
    row = bias[0, 2:].reshape(4, 8)
    assert torch.allclose(row[:, :4], torch.full((4, 4), math.log(0.1)))  # agentview damped
    assert torch.all(row[:, 4:] == 0)                    # wrist tokens free


def test_phase_aware_bias_only_when_gripper_open():
    bias = wrist_attention_bias(50, 8, [4, 4], 4, 1, 0.1, device='cpu', dtype=torch.float32,
                                gripper_open=torch.tensor([True, False]), num_heads=8)
    assert bias.shape == (16, 50, 34)
    assert bias[:8].min() < 0          # open gripper: bias applied
    assert torch.all(bias[8:] == 0)    # closed gripper: no bias


def test_phase_aware_policy_runs():
    policy = make('pa_act_phase')
    policy.set_observation_normalization(torch.zeros(9), torch.ones(9))
    qpos, images, actions, pad = fake_batch(50)
    assert torch.isfinite(policy(qpos, images, actions, pad)['loss'])
