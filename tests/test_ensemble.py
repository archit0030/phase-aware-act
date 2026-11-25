import math

import torch

from pa_act.temporal import TemporalEnsembler


def test_static_weights_follow_age():
    ens = TemporalEnsembler(5, m=0.5)
    for t in range(3):
        r = ens.update(t, torch.full((5, 7), float(t)))
    # three chunks of age 2, 1, 0
    expected = torch.softmax(torch.tensor([-1.0, -0.5, 0.0], dtype=torch.float64), 0)
    assert torch.allclose(r.weights, expected)
    assert math.isclose(float(r.effective_horizon), float(1 / (expected ** 2).sum()))


def test_old_chunks_drop_out():
    ens = TemporalEnsembler(3)
    for t in range(10):
        r = ens.update(t, torch.zeros(3, 7))
    assert r.ages.tolist() == [2, 1, 0]


def test_dute_downweights_disagreeing_chunk():
    static = TemporalEnsembler(4, m=0.01)
    dute = TemporalEnsembler(4, m=0.01, use_dute=True, lambda_2=1.0)
    old = torch.zeros(4, 7)
    old[:, 0] = 3.0                      # old chunk disagrees in x
    new = torch.zeros(4, 7)
    for ens in (static, dute):
        ens.update(0, old)
    w_static = static.update(1, new).weights[0]
    w_dute = dute.update(1, new).weights[0]
    assert w_dute < w_static


def test_gripper_not_used_in_gates():
    dute = TemporalEnsembler(4, use_dute=True, lambda_2=1.0)
    old = torch.zeros(4, 7)
    old[:, 6] = 5.0                      # disagreement only in the gripper
    dute.update(0, old)
    r = dute.update(1, torch.zeros(4, 7))
    assert float(r.divergence_l2[0]) == 0.0


def test_uncertainty_gate():
    ens = TemporalEnsembler(4, use_dute=True, lambda_1=1.0)
    ens.update(0, torch.zeros(4, 7), torch.full((4, 7), 2.0))   # unsure old chunk
    r = ens.update(1, torch.zeros(4, 7), torch.full((4, 7), 0.5))
    assert r.weights[1] > r.weights[0]
