"""Mathematical verification that equivariance error is zero for a truly equivariant op."""

import torch
import pytest
from src.pino.fourier_derivatives import spectral_grad, spectral_laplacian


def test_spectral_grad_shape():
    f = torch.randn(2, 1, 32, 32)
    df_dx, df_dy = spectral_grad(f)
    assert df_dx.shape == f.shape
    assert df_dy.shape == f.shape


def test_spectral_grad_sine():
    """For f(x,y) = sin(2πx), ∂f/∂x = 2π·cos(2πx)."""
    nx, ny = 64, 64
    x = torch.linspace(0, 1, nx, dtype=torch.float32)
    y = torch.linspace(0, 1, ny, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    f = torch.sin(2 * torch.pi * X).unsqueeze(0).unsqueeze(0)
    df_dx, _ = spectral_grad(f)
    expected = (2 * torch.pi * torch.cos(2 * torch.pi * X)).unsqueeze(0).unsqueeze(0)
    # Should match to within 1% relative error (periodic boundary)
    rel_err = (df_dx - expected).norm() / (expected.norm() + 1e-8)
    assert rel_err.item() < 0.01, f"Spectral grad relative error = {rel_err:.4f}"


def test_spectral_laplacian_shape():
    f = torch.randn(2, 1, 32, 32)
    lap = spectral_laplacian(f)
    assert lap.shape == f.shape
