"""PyTorch Dataset for loading processed flow-field data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    """Load normalized flow fields from a processed dataset.npz.

    Inputs are permuted from (N, H, W, 3) → (N, 3, H, W) to match
    PyTorch's channels-first convention expected by FNO.

    Args:
        data_path: Path to data/processed/dataset.npz.
        split_idx_path: Path to a split index file (train/val/test_idx.npy).
    """

    def __init__(self, data_path: str, split_idx_path: str):
        data = np.load(data_path)
        idx = np.load(split_idx_path)
        # (N, H, W, 3) → (N, 3, H, W) channels-first for PhysicsNeMo FNO
        self.inputs  = torch.from_numpy(data["inputs"][idx]).permute(0, 3, 1, 2)
        self.targets = torch.from_numpy(data["targets"][idx]).permute(0, 3, 1, 2)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
