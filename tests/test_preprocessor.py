"""Tests for DataPreprocessor: raw .npz → processed dataset + splits."""

import numpy as np
import pytest
from pathlib import Path


def _make_fake_raw_dir(tmp_path: Path, n: int = 6) -> Path:
    """Create synthetic raw .npz files for testing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    H, W = 16, 16
    rng = np.random.default_rng(0)
    re_vals = [50.0, 100.0, 150.0, 50.0, 100.0, 150.0]
    for k in range(n):
        re = re_vals[k]
        cx, cy = 0.4 + 0.05 * (k % 2), 0.5
        mask = np.zeros((H, W), dtype=bool)
        mask[7:9, 5:7] = True
        np.savez(
            raw_dir / f"re{re:.1f}_cx{cx:.3f}_cy{cy:.3f}.npz",
            geom_mask=mask, re=re, cx=cx, cy=cy,
            u=rng.uniform(-1, 2, (H, W)).astype(np.float32),
            v=rng.uniform(-1, 1, (H, W)).astype(np.float32),
            p=rng.uniform(-0.5, 0.5, (H, W)).astype(np.float32),
        )
    return raw_dir


def test_preprocessor_output_shapes(tmp_path):
    from src.data_gen.preprocessor import DataPreprocessor
    raw_dir = _make_fake_raw_dir(tmp_path, n=6)
    proc = DataPreprocessor(
        raw_dir=str(raw_dir),
        processed_dir=str(tmp_path / "processed"),
        splits_dir=str(tmp_path / "splits"),
        re_min=20.0, re_max=200.0,
    )
    proc.process()
    data = np.load(tmp_path / "processed" / "dataset.npz")
    assert data["inputs"].shape == (6, 16, 16, 3)
    assert data["targets"].shape == (6, 16, 16, 3)


def test_preprocessor_splits_cover_all(tmp_path):
    from src.data_gen.preprocessor import DataPreprocessor
    raw_dir = _make_fake_raw_dir(tmp_path, n=6)
    proc = DataPreprocessor(
        raw_dir=str(raw_dir),
        processed_dir=str(tmp_path / "processed"),
        splits_dir=str(tmp_path / "splits"),
        re_min=20.0, re_max=200.0,
    )
    proc.process()
    train = np.load(tmp_path / "splits" / "train_idx.npy")
    val   = np.load(tmp_path / "splits" / "val_idx.npy")
    test  = np.load(tmp_path / "splits" / "test_idx.npy")
    all_idx = np.concatenate([train, val, test])
    assert len(all_idx) == 6
    assert len(set(all_idx.tolist())) == 6


def test_preprocessor_re_channel_uniform(tmp_path):
    from src.data_gen.preprocessor import DataPreprocessor
    raw_dir = _make_fake_raw_dir(tmp_path, n=6)
    proc = DataPreprocessor(
        raw_dir=str(raw_dir),
        processed_dir=str(tmp_path / "processed"),
        splits_dir=str(tmp_path / "splits"),
        re_min=20.0, re_max=200.0,
    )
    proc.process()
    inputs = np.load(tmp_path / "processed" / "dataset.npz")["inputs"]
    for n in range(len(inputs)):
        re_ch = inputs[n, :, :, 1]
        assert 0.0 <= re_ch[0, 0] <= 1.0
        assert np.allclose(re_ch, re_ch[0, 0]), "Re channel must be spatially uniform"


def test_preprocessor_inlet_channel(tmp_path):
    from src.data_gen.preprocessor import DataPreprocessor
    raw_dir = _make_fake_raw_dir(tmp_path, n=6)
    proc = DataPreprocessor(
        raw_dir=str(raw_dir),
        processed_dir=str(tmp_path / "processed"),
        splits_dir=str(tmp_path / "splits"),
        re_min=20.0, re_max=200.0,
    )
    proc.process()
    inlet_ch = np.load(tmp_path / "processed" / "dataset.npz")["inputs"][0, :, :, 2]
    assert np.allclose(inlet_ch[:, 0], 1.0), "Inlet column (j=0) should be 1.0"
    assert np.allclose(inlet_ch[:, 1:], 0.0), "Non-inlet columns should be 0.0"
