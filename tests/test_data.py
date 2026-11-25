import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pa_act.data import LiberoChunkDataset, inspect_demonstrations, normalization_stats

CFG = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'act_k10.json').read_text())['policy']


@pytest.fixture
def demo_file(tmp_path):
    path = tmp_path / 'task_demo.hdf5'
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        for i in range(3):
            T = 20 + i
            g = f.create_group(f'data/demo_{i}')
            g['actions'] = rng.normal(size=(T, 7))
            g['obs/joint_states'] = rng.normal(size=(T, 7))
            g['obs/gripper_states'] = rng.normal(size=(T, 2))
            g['obs/agentview_rgb'] = rng.integers(0, 255, (T, 128, 128, 3), dtype=np.uint8)
            g['obs/eye_in_hand_rgb'] = rng.integers(0, 255, (T, 128, 128, 3), dtype=np.uint8)
    return path


def test_dataset_shapes_and_padding(demo_file):
    keys = inspect_demonstrations(demo_file, CFG, required=3)
    stats = normalization_stats(demo_file, keys, CFG)
    ds = LiberoChunkDataset(demo_file, keys, CFG, stats)
    # action offset 1: one step fewer per demo
    assert len(ds) == 19 + 20 + 21
    q, img, act, pad = ds[len(ds) - 1]          # last step of the last demo
    assert q.shape == (9,) and img.shape == (2, 3, 128, 128) and act.shape == (10, 7)
    assert 0 <= float(img.min()) and float(img.max()) <= 1
    assert pad[1:].all() and not pad[0]
    assert (act[pad] == 0).all()


def test_normalisation_has_floor(demo_file):
    keys = inspect_demonstrations(demo_file, CFG, required=3)
    stats = normalization_stats(demo_file, keys, CFG, floor=10.0)
    assert (stats['action_std'] == 10.0).all()


def test_too_few_demos(demo_file):
    with pytest.raises(ValueError):
        inspect_demonstrations(demo_file, CFG, required=50)
