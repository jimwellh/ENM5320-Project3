"""Fourier spectral differentiation — consistent with course L3 spectral methods.

Computes spatial derivatives by multiplying in frequency space (wavenumber * i),
then transforming back. More accurate than finite differences for smooth periodic fields.
"""

import torch


def spectral_grad(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ∂f/∂x and ∂f/∂y via Fourier spectral differentiation.

    Args:
        f: Field tensor of shape (batch, 1, H, W), assumed periodic.

    Returns:
        Tuple (df_dx, df_dy), each shape (batch, 1, H, W).
    """
    batch, c, ny, nx = f.shape
    F = torch.fft.rfft2(f)

    kx = torch.fft.rfftfreq(nx, d=1.0 / nx).to(f.device)
    ky = torch.fft.fftfreq(ny, d=1.0 / ny).to(f.device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    KX = KX.unsqueeze(0).unsqueeze(0)
    KY = KY.unsqueeze(0).unsqueeze(0)

    df_dx = torch.fft.irfft2(1j * 2 * torch.pi * KX * F, s=(ny, nx))
    df_dy = torch.fft.irfft2(1j * 2 * torch.pi * KY * F, s=(ny, nx))
    return df_dx, df_dy


def spectral_laplacian(f: torch.Tensor) -> torch.Tensor:
    """Compute ∇²f = ∂²f/∂x² + ∂²f/∂y² via Fourier spectral differentiation.

    Args:
        f: Field tensor of shape (batch, 1, H, W).

    Returns:
        Laplacian, shape (batch, 1, H, W).
    """
    batch, c, ny, nx = f.shape
    F = torch.fft.rfft2(f)

    kx = torch.fft.rfftfreq(nx, d=1.0 / nx).to(f.device)
    ky = torch.fft.fftfreq(ny, d=1.0 / ny).to(f.device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    KX = KX.unsqueeze(0).unsqueeze(0)
    KY = KY.unsqueeze(0).unsqueeze(0)

    lap = torch.fft.irfft2(-(2 * torch.pi) ** 2 * (KX ** 2 + KY ** 2) * F, s=(ny, nx))
    return lap
