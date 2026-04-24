"""Standard FNO baseline using PhysicsNeMo's spectral encoder building block.

Imports from physicsnemo.nn directly to avoid DiT's distributed-tensor dependency,
which is incompatible with PyTorch >= 2.9 (torch.distributed.tensor._ops.registration
was removed).  Architecture: FNO2DEncoder (spectral layers) + pixel-wise MLP decoder.
"""

import torch
import torch.nn as nn


class FNOBaseline(nn.Module):
    """FNO for (geometry, Re) → (u, v, p) prediction.

    Input:  (batch, 3, H, W) — channels: [geom_mask, Re_normalized, inlet_profile]
    Output: (batch, 3, H, W) — channels: [u, v, p]

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to keep per dimension.
        width: Latent channel dimension for spectral layers.
        n_layers: Number of FNO spectral layers.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 modes: int = 16, width: int = 64, n_layers: int = 4):
        super().__init__()
        try:
            from physicsnemo.nn.module.fno_layers import FNO2DEncoder
        except ImportError:
            raise ImportError("Install nvidia-physicsnemo: pip install nvidia-physicsnemo")

        self.encoder = FNO2DEncoder(
            in_channels=in_channels,
            num_fno_layers=n_layers,
            fno_layer_size=width,
            num_fno_modes=modes,
        )
        # Pixel-wise MLP decoder: (width) → 128 → out_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
