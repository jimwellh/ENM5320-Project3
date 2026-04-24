"""Evaluation metrics: relative L2 error and equivariance error."""

import torch
import torch.nn as nn
import numpy as np


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean per-sample relative L2 error.

    Computes ||ŷ_n - y_n||₂ / ||y_n||₂ for each sample n, then averages
    over the batch. This prevents high-Re samples from dominating the metric.

    Args:
        pred: (N, C, H, W) predicted fields.
        target: (N, C, H, W) ground-truth fields.

    Returns:
        Mean per-sample relative L2 error (scalar float).
    """
    diff_norm   = torch.linalg.vector_norm(pred - target, dim=(1, 2, 3))        # (N,)
    target_norm = torch.linalg.vector_norm(target,        dim=(1, 2, 3)) + 1e-8  # (N,)
    return (diff_norm / target_norm).mean().item()


def equivariance_error(model: nn.Module, x: torch.Tensor,
                       rotation_fn, device: str = "cuda") -> float:
    """Measure how equivariant the model is to a spatial transformation.

    Compares f(T(x)) vs T(f(x)) — should be zero for a perfectly equivariant model.

    Args:
        model: Neural operator.
        x: Input batch (N, C, H, W).
        rotation_fn: Function that rotates a field tensor.
        device: Device string.

    Returns:
        Mean relative equivariance error across the batch.
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        pred_original = model(x)
        x_rotated = rotation_fn(x)
        pred_rotated = model(x_rotated)
        pred_original_rotated = rotation_fn(pred_original)
        err = relative_l2_error(pred_rotated, pred_original_rotated)
    return err


if __name__ == "__main__":
    import argparse
    import torch
    from pathlib import Path
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.models.model_factory import build_model
    from src.training.dataset import FlowDataset

    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--model",      required=True,
                        choices=["fno_baseline", "equivariant_fno"],
                        help="Model type (must match config file name)")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    cli = parser.parse_args()

    cfg = OmegaConf.load(f"configs/{cli.model}.yaml")

    model = build_model(
        cfg.model_type,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        modes=cfg.fno_modes,
        width=cfg.fno_width,
        n_layers=cfg.n_layers,
    )
    state_dict = torch.load(cli.checkpoint, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model = model.to(cfg.device)
    model.eval()

    test_ds = FlowDataset("data/processed/dataset.npz", "data/splits/test_idx.npy")
    loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(cfg.device)).cpu())
            targets.append(y)
    preds_cat   = torch.cat(preds)    # (N, 3, H, W) — normalized space
    targets_cat = torch.cat(targets)

    # Denormalize: restore physical units before computing relative L2
    # stats channels: [u, v, p] matching target channel order
    stats = np.load("data/processed/stats.npz")
    means = torch.tensor(
        [float(stats["u_mean"]), float(stats["v_mean"]), float(stats["p_mean"])],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    stds = torch.tensor(
        [float(stats["u_std"]), float(stats["v_std"]), float(stats["p_std"])],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    preds_phys   = preds_cat   * stds + means
    targets_phys = targets_cat * stds + means

    err = relative_l2_error(preds_phys, targets_phys)
    print(f"Test relative L2 error (physical space): {err:.6f} ({err * 100:.2f}%)")
    print(f"Target < 5.00% → {'PASS' if err < 0.05 else 'FAIL'}")
