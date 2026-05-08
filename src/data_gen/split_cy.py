"""Splits dataset based on cylinder y-coordinate for generalization experiment."""

import numpy as np
from pathlib import Path
import random

def create_cy_splits(raw_dir: str = "data/raw", splits_dir: str = "data/splits", threshold: float = 0.5):
    raw_path = Path(raw_dir)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    
    paths = sorted(raw_path.glob("*.npz"))
    if not paths:
        print(f"No .npz files found in {raw_dir}")
        return

    lower_half_idx = []
    upper_half_idx = []

    for i, p in enumerate(paths):
        d = np.load(p)
        cy = float(d["cy"])
        if cy <= threshold:
            lower_half_idx.append(i)
        else:
            upper_half_idx.append(i)

    print(f"Total samples: {len(paths)}")
    print(f"Lower half (cy <= {threshold}): {len(lower_half_idx)}")
    print(f"Upper half (cy > {threshold}): {len(upper_half_idx)}")

    # Split lower half into train (80%) and val (20%)
    rng = random.Random(42)
    rng.shuffle(lower_half_idx)
    
    n_train = int(len(lower_half_idx) * 0.8)
    train_idx = lower_half_idx[:n_train]
    val_idx = lower_half_idx[n_train:]
    
    # Upper half will be test
    test_idx = upper_half_idx
    
    # Save splits
    np.save(splits_path / "train_cy_lower.npy", np.array(train_idx))
    np.save(splits_path / "val_cy_lower.npy", np.array(val_idx))
    np.save(splits_path / "test_cy_upper.npy", np.array(test_idx))
    
    print(f"Saved train_cy_lower.npy ({len(train_idx)} samples)")
    print(f"Saved val_cy_lower.npy ({len(val_idx)} samples)")
    print(f"Saved test_cy_upper.npy ({len(test_idx)} samples)")

if __name__ == "__main__":
    create_cy_splits()
