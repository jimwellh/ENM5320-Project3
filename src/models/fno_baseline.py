"""Standard FNO baseline using PhysicsNeMo's production implementation."""

import torch
import torch.nn as nn


class FNOBaseline(nn.Module):
    """Wraps PhysicsNeMo FNO for (geometry, Re) → (u, v, p) prediction.

    Input:  (batch, 3, H, W) — channels: [geom_mask, Re_normalized, inlet_profile]
    Output: (batch, 3, H, W) — channels: [u, v, p]

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to keep per dimension.
        width: Lifting channel dimension.
        n_layers: Number of FNO layers.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 modes: int = 16, width: int = 64, n_layers: int = 4):
        super().__init__()
        try:
            from physicsnemo.models.fno import FNO
            self.model = FNO(
                in_channels=in_channels,
                out_channels=out_channels,
                decoder_layers=1,
                decoder_layer_size=128,
                dimension=2,
                latent_channels=width,
                num_fno_layers=n_layers,
                num_fno_modes=modes,
                padding=9,
            )
        except ImportError:
            raise ImportError("Install nvidia-physicsnemo: pip install nvidia-physicsnemo")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
