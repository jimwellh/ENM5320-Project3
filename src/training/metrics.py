"""Evaluation metrics: relative L2 error and equivariance error."""

import torch
import torch.nn as nn
import numpy as np


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute relative L2 error between predicted and target fields.

    Args:
        pred: (N, C, H, W) predicted fields.
        target: (N, C, H, W) ground-truth fields.

    Returns:
        Scalar relative L2 error.
    """
    diff = pred - target
    return (torch.norm(diff) / (torch.norm(target) + 1e-8)).item()


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
