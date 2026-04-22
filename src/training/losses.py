"""Loss functions: data-driven L2 and physics-informed NS residual (PINO)."""

import torch
import torch.nn as nn


class DataLoss(nn.Module):
    """Relative L2 loss between predicted and ground-truth fields."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                inputs: torch.Tensor | None = None) -> torch.Tensor:
        diff = pred - target
        return torch.norm(diff) / (torch.norm(target) + 1e-8)


class PINOLoss(nn.Module):
    """PINO loss: data loss + weighted Navier-Stokes residual.

    Uses Fourier spectral derivatives (consistent with course L3 spectral differentiation).

    Args:
        pde_weight: Weight for the NS residual term (lambda).
    """

    def __init__(self, pde_weight: float = 0.1):
        super().__init__()
        self.pde_weight = pde_weight
        self.data_loss = DataLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                inputs: torch.Tensor) -> torch.Tensor:
        loss_data = self.data_loss(pred, target)
        loss_pde = self._ns_residual(pred, inputs)
        return loss_data + self.pde_weight * loss_pde

    def _ns_residual(self, pred: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute NS residual via Fourier spectral derivatives.

        Args:
            pred: (batch, 3, H, W) — [u, v, p]
            inputs: (batch, 3, H, W) — [geom_mask, Re_normalized, inlet]
        """
        from src.pino.fourier_derivatives import spectral_grad
        u = pred[:, 0:1]
        v = pred[:, 1:2]
        p = pred[:, 2:3]
        re = inputs[:, 1:2, 0, 0] * 180.0 + 20.0  # denormalize Re

        du_dx, du_dy = spectral_grad(u)
        dv_dx, dv_dy = spectral_grad(v)
        dp_dx, _ = spectral_grad(p)
        _, dp_dy = spectral_grad(p)

        # Continuity residual: ∇·u = 0
        continuity = du_dx + dv_dy

        # Momentum residuals (simplified steady-state)
        re = re[:, :, None, None]
        mom_x = u * du_dx + v * du_dy + dp_dx - (1.0 / re) * (
            spectral_grad(du_dx)[0] + spectral_grad(du_dy)[1]
        )
        mom_y = u * dv_dx + v * dv_dy + dp_dy - (1.0 / re) * (
            spectral_grad(dv_dx)[0] + spectral_grad(dv_dy)[1]
        )

        residual = continuity.pow(2).mean() + mom_x.pow(2).mean() + mom_y.pow(2).mean()
        return residual
