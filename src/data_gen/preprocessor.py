"""Converts raw solver .npz snapshots into normalized training dataset."""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


class DataPreprocessor:
    """Builds processed dataset.npz and train/val/test splits from raw .npz files.

    Input channels [geom_mask_float, Re_normalized, inlet_profile]:
      - Ch 0: solid mask (0.0=fluid, 1.0=solid)
      - Ch 1: Re_D normalized to [0,1], spatially uniform
      - Ch 2: inlet profile (1.0 at j=0, 0.0 elsewhere)

    Target channels [u_norm, v_norm, p_norm]:
      - Z-score normalized by training-set statistics (stats.npz)

    Args:
        raw_dir: Directory with per-simulation .npz files.
        processed_dir: Output for dataset.npz and stats.npz.
        splits_dir: Output for train/val/test index arrays (.npy).
        re_min, re_max: Dataset-level Re_D bounds for normalization.
        train_ratio: Fraction of samples for training (default 0.70).
        val_ratio: Fraction of samples for validation (default 0.15).
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        splits_dir: str = "data/splits",
        re_min: float = 20.0,
        re_max: float = 200.0,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.splits_dir = Path(splits_dir)
        self.re_min = re_min
        self.re_max = re_max
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def process(self) -> None:
        """Run full preprocessing: load raw → normalize → split → save."""
        samples = self._load_raw()
        N = len(samples)
        if N == 0:
            raise RuntimeError(f"No .npz files found in {self.raw_dir}")

        rng = np.random.default_rng(42)
        perm = rng.permutation(N)
        n_train = int(N * self.train_ratio)
        n_val = int(N * self.val_ratio)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        stats = self._compute_stats(samples, train_idx)
        inputs = self._build_inputs(samples, stats)    # (N, H, W, 3)
        targets = self._build_targets(samples, stats)  # (N, H, W, 3)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self.processed_dir / "dataset.npz", inputs=inputs, targets=targets)
        np.savez(self.processed_dir / "stats.npz", **stats)

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.splits_dir / "train_idx.npy", train_idx)
        np.save(self.splits_dir / "val_idx.npy", val_idx)
        np.save(self.splits_dir / "test_idx.npy", test_idx)

        print(f"Processed {N} samples → train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")

    def _load_raw(self) -> list[dict]:
        paths = sorted(self.raw_dir.glob("*.npz"))
        samples = []
        for p in paths:
            d = np.load(p)
            samples.append({
                "geom_mask": d["geom_mask"],
                "re": float(d["re"]),
                "u": d["u"].astype(np.float32),
                "v": d["v"].astype(np.float32),
                "p": d["p"].astype(np.float32),
            })
        return samples

    def _compute_stats(self, samples: list[dict], train_idx: np.ndarray) -> dict:
        eps = 1e-8
        u_all = np.stack([samples[i]["u"] for i in train_idx])
        v_all = np.stack([samples[i]["v"] for i in train_idx])
        p_all = np.stack([samples[i]["p"] for i in train_idx])
        return {
            "u_mean": float(u_all.mean()), "u_std": float(u_all.std() + eps),
            "v_mean": float(v_all.mean()), "v_std": float(v_all.std() + eps),
            "p_mean": float(p_all.mean()), "p_std": float(p_all.std() + eps),
            "re_min": self.re_min, "re_max": self.re_max,
        }

    def _build_inputs(self, samples: list[dict], stats: dict) -> np.ndarray:
        N = len(samples)
        H, W = samples[0]["geom_mask"].shape
        inputs = np.zeros((N, H, W, 3), dtype=np.float32)
        re_range = stats["re_max"] - stats["re_min"]

        for n, s in enumerate(samples):
            inputs[n, :, :, 0] = s["geom_mask"].astype(np.float32)
            inputs[n, :, :, 1] = (s["re"] - stats["re_min"]) / re_range

        # Channel 2: inlet profile — 1.0 at j=0, 0.0 elsewhere (broadcast across N)
        inlet = np.zeros((H, W), dtype=np.float32)
        inlet[:, 0] = 1.0
        inputs[:, :, :, 2] = inlet[np.newaxis]
        return inputs

    def _build_targets(self, samples: list[dict], stats: dict) -> np.ndarray:
        N = len(samples)
        H, W = samples[0]["u"].shape
        targets = np.zeros((N, H, W, 3), dtype=np.float32)
        for n, s in enumerate(samples):
            targets[n, :, :, 0] = (s["u"] - stats["u_mean"]) / stats["u_std"]
            targets[n, :, :, 1] = (s["v"] - stats["v_mean"]) / stats["v_std"]
            targets[n, :, :, 2] = (s["p"] - stats["p_mean"]) / stats["p_std"]
        return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw CFD snapshots into training data.")
    parser.add_argument("--raw_dir",       default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--splits_dir",    default="data/splits")
    args = parser.parse_args()
    DataPreprocessor(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        splits_dir=args.splits_dir,
    ).process()
