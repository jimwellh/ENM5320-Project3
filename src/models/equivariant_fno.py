"""SE(2)-equivariant FNO variant using group-equivariant convolutions (e2cnn).

Strategy: replace the pointwise lifting/projection layers in FNO with steerable
convolutions from e2cnn, keeping the Fourier layers in the frequency domain.
Equivariance group: p4 (discrete 90-degree rotations) as a first step.

Reference: Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs" (NeurIPS 2019).
Install: pip install e2cnn
"""

import torch
import torch.nn as nn


class EquivariantFNO(nn.Module):
    """FNO with SE(2)-equivariant lifting and projection layers.

    Input:  (batch, 3, H, W)
    Output: (batch, 3, H, W)

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Fourier modes per dimension.
        width: Equivariant feature field width.
        n_layers: Number of FNO spectral layers.
        group: Symmetry group — 'p4' (90° rotations) or 'p4m' (rotations + reflections).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 modes: int = 16, width: int = 64, n_layers: int = 4,
                 group: str = "p4"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.group = group
        # Full implementation in Milestone 3
        raise NotImplementedError(
            "Equivariant FNO: implement group convolution lifting in Milestone 3. "
            "Install e2cnn first: pip install e2cnn"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
