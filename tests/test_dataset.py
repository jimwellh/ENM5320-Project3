"""Tests for FlowDataset."""

import numpy as np
import torch
from pathlib import Path


def _make_synthetic_npz(tmp_path: Path, N: int = 20, H: int = 16, W: int = 16):
    rng = np.random.default_rng(0)
    data_path = tmp_path / "dataset.npz"
    idx_path  = tmp_path / "train_idx.npy"
    np.savez(
        data_path,
        inputs=rng.random((N, H, W, 3)).astype(np.float32),
        targets=rng.random((N, H, W, 3)).astype(np.float32),
    )
    idx = np.arange(N // 2)
    np.save(idx_path, idx)
    return str(data_path), str(idx_path), len(idx)


def test_flow_dataset_length(tmp_path):
    from src.training.dataset import FlowDataset
    data_path, idx_path, n = _make_synthetic_npz(tmp_path)
    ds = FlowDataset(data_path, idx_path)
    assert len(ds) == n


def test_flow_dataset_tensor_shapes(tmp_path):
    from src.training.dataset import FlowDataset
    data_path, idx_path, _ = _make_synthetic_npz(tmp_path, N=20, H=16, W=16)
    ds = FlowDataset(data_path, idx_path)
    x, y = ds[0]
    assert x.shape == (3, 16, 16), f"Expected (3,16,16), got {x.shape}"
    assert y.shape == (3, 16, 16), f"Expected (3,16,16), got {y.shape}"


def test_flow_dataset_dtype(tmp_path):
    from src.training.dataset import FlowDataset
    data_path, idx_path, _ = _make_synthetic_npz(tmp_path)
    ds = FlowDataset(data_path, idx_path)
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_flow_dataset_index_selection(tmp_path):
    from src.training.dataset import FlowDataset
    N, H, W = 10, 8, 8
    rng = np.random.default_rng(1)
    data_path = tmp_path / "dataset.npz"
    idx_path  = tmp_path / "idx.npy"
    inputs_np  = rng.random((N, H, W, 3)).astype(np.float32)
    targets_np = rng.random((N, H, W, 3)).astype(np.float32)
    np.savez(data_path, inputs=inputs_np, targets=targets_np)
    selected = np.array([0, 3, 7])
    np.save(idx_path, selected)

    ds = FlowDataset(str(data_path), str(idx_path))
    assert len(ds) == 3

    x0, _ = ds[0]
    expected = torch.from_numpy(inputs_np[0]).permute(2, 0, 1)
    assert torch.allclose(x0, expected), "ds[0] should match inputs_np[selected[0]]"
