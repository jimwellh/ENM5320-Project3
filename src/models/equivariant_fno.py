"""SE(2)-equivariant FNO variant using group-equivariant convolutions (escnn).

Strategy: replace the pointwise lifting/projection layers in FNO with steerable
convolutions from escnn, keeping the Fourier layers in the frequency domain.
Equivariance group: p4 (discrete 90-degree rotations).

Reference: Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs" (NeurIPS 2019).
Library: escnn 1.0.11 — https://github.com/QUVA-Lab/escnn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as enn


class SpectralConv2d(nn.Module):
    """2D Fourier spectral convolution (FNO-style) on raw (B, C, H, W) tensors.

    Multiplies the lowest `modes x modes` Fourier coefficients by learned
    complex weights and zero-pads the rest before the inverse FFT.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        modes:        Number of Fourier modes kept per spatial dimension.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes,
                                dtype=torch.cfloat)
        )
        self.modes = modes
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float tensor.
        Returns:
            (B, out_channels, H, W) float tensor.
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, s=(H, W))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m = self.modes
        out_ft[:, :, :m, :m] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m, :m], self.weights
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class EquivariantFNOBlock(nn.Module):
    """Single FNO block with equivariant pointwise path + spectral path.

    Path A (equivariant): 1×1 G-conv that respects the p4 group action.
    Path B (spectral):    extract `.tensor`, apply SpectralConv2d, wrap back.
    Output: ReLU(A + B) wrapped as GeometricTensor.

    Args:
        feat_type: escnn FieldType for both input and output.
        modes:     Fourier modes for the spectral path.
    """

    def __init__(self, feat_type: enn.FieldType, modes: int):
        super().__init__()
        self.feat_type = feat_type
        self.equi_conv = enn.R2Conv(feat_type, feat_type, kernel_size=1,
                                    bias=False)
        channels = feat_type.size
        self.spectral_conv = SpectralConv2d(channels, channels, modes)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        # Path A — equivariant 1×1 conv
        path_a = self.equi_conv(x)
        # Path B — frequency-domain conv on raw tensor
        path_b = self.spectral_conv(x.tensor)
        # Merge and re-wrap
        result = F.relu(path_a.tensor + path_b)
        return enn.GeometricTensor(result, self.feat_type)


class EquivariantFNO(nn.Module):
    """FNO with SE(2)-equivariant lifting and projection layers.

    Architecture:
        lift    : R2Conv(feat_in  → feat_hidden, 3×3)   — scalar → regular field
        blocks  : n_layers × EquivariantFNOBlock          — equi + spectral
        proj1   : R2Conv(feat_hidden → feat_half, 1×1)   — compress
        proj2   : R2Conv(feat_half  → feat_out,  1×1)    — [u,v] + [p]

    Feature type sizes (p4 group, regular_repr has size 4):
        feat_in     : in_channels  trivial reprs  → size = in_channels
        feat_hidden : width        regular reprs  → size = width * 4
        feat_half   : width//2     regular reprs  → size = (width//2) * 4
        feat_out    : [irrep(1), trivial_repr]     → size = 3  (u,v vector + p scalar)

    Input:  (batch, 3, H, W)
    Output: (batch, 3, H, W)

    Args:
        in_channels:  Number of input channels (must match input tensor channels).
        out_channels: Number of output channels (must be 3 — [u, v, p]).
        modes:        Fourier modes per dimension.
        width:        Equivariant feature field width (number of regular reprs).
        n_layers:     Number of EquivariantFNOBlock layers.
        group:        Symmetry group string ('p4' supported; ignored if not 'p4').
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 modes: int = 16, width: int = 64, n_layers: int = 4,
                 group: str = "p4"):
        super().__init__()
        assert out_channels == 3, (
            f"EquivariantFNO requires out_channels=3 (got {out_channels}). "
            "The output feature type encodes [u, v] as irrep(1) and [p] as "
            "trivial_repr, which together have exactly 3 channels."
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # --- Group space (p4: discrete 90° rotations in 2D) ---
        g = gspaces.rot2dOnR2(N=4)

        # --- Feature types ---
        feat_in = enn.FieldType(g, in_channels * [g.trivial_repr])
        # regular_repr has size 4 for p4, so feat_hidden.size = width * 4
        feat_hidden = enn.FieldType(g, width * [g.regular_repr])
        # feat_half.size = (width // 2) * 4
        feat_half = enn.FieldType(g, (width // 2) * [g.regular_repr])
        # irrep(1) is the standard 2D rotation irrep (size=2) for vector fields
        # trivial_repr (size=1) for the scalar pressure field
        # Together: size = 3
        feat_out = enn.FieldType(g, [g.irrep(1), g.trivial_repr])

        # Store feat types needed in forward()
        self.feat_in = feat_in
        self.feat_hidden = feat_hidden

        # --- Layers ---
        # Lifting: (B, in_channels, H, W) → (B, width*4, H, W)
        self.lift = enn.R2Conv(feat_in, feat_hidden,
                               kernel_size=3, padding=1, bias=False)

        # Equivariant FNO blocks
        self.blocks = nn.ModuleList([
            EquivariantFNOBlock(feat_hidden, modes) for _ in range(n_layers)
        ])

        # Projection: (B, width*4, H, W) → (B, width*2, H, W)
        self.proj1 = enn.R2Conv(feat_hidden, feat_half,
                                kernel_size=1, bias=False)
        # Projection: (B, width*2, H, W) → (B, 3, H, W)
        self.proj2 = enn.R2Conv(feat_half, feat_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) float tensor.
        Returns:
            (B, 3, H, W) float tensor — predicted [u, v, p] fields.
        """
        # Wrap as equivariant tensor
        gx = enn.GeometricTensor(x, self.feat_in)

        # Lift to hidden equivariant feature field
        h = self.lift(gx)

        # Apply FNO blocks
        for block in self.blocks:
            h = block(h)

        # Project to output
        h = self.proj1(h)
        h = self.proj2(h)

        # Return raw tensor (B, 3, H, W)
        return h.tensor
