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
    # Periodic domain [0, 1): exclude endpoint so FFT assumes dx = 1/N
    x = torch.arange(nx, dtype=torch.float32) / nx
    y = torch.arange(ny, dtype=torch.float32) / ny
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


# ══════════════════════════════════════════════════════════════════════════════
# Tensor rotation utilities (local to this module)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
from pathlib import Path
from scipy.ndimage import rotate as _scipy_rotate


def rotate_input_tensor(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate all C channels of (B, C, H, W) spatially using scipy (CCW convention)."""
    x_np = x.cpu().numpy()
    result = np.stack([
        np.stack([
            _scipy_rotate(x_np[b, c], angle_deg, reshape=False)
            for c in range(x_np.shape[1])
        ])
        for b in range(x_np.shape[0])
    ])
    return torch.tensor(result, dtype=x.dtype, device=x.device)


def rotate_output_tensor(y: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate flow field (B, 3, H, W)=[u,v,p] with vector-aware transform.

    Calls rotate_flow_field so that (u,v) components undergo the 2-D rotation
    matrix in addition to the spatial grid rotation.
    """
    from src.data_gen.geometry import rotate_flow_field
    y_np = y.cpu().numpy()
    results = []
    for b in range(y_np.shape[0]):
        u_r, v_r, p_r = rotate_flow_field(y_np[b, 0], y_np[b, 1], y_np[b, 2], angle_deg)
        results.append(np.stack([u_r, v_r, p_r]))
    return torch.tensor(np.stack(results), dtype=y.dtype, device=y.device)


# ══════════════════════════════════════════════════════════════════════════════
# Tests for rotation utilities
# ══════════════════════════════════════════════════════════════════════════════

def test_rotate_input_tensor_shape():
    x = torch.randn(2, 3, 64, 64)
    x_rot = rotate_input_tensor(x, 90.0)
    assert x_rot.shape == x.shape


def test_rotate_input_tensor_identity():
    x = torch.randn(2, 3, 64, 64)
    x_rot = rotate_input_tensor(x, 0.0)
    assert torch.allclose(x_rot, x, atol=1e-5)


def test_rotate_output_tensor_rightward_to_upward():
    """90° CCW: uniform rightward flow (u=1, v=0) → upward flow (u=0, v=1)."""
    H, W = 64, 64
    u = torch.ones(1, H, W)
    v = torch.zeros(1, H, W)
    p = torch.zeros(1, H, W)
    y = torch.stack([u, v, p], dim=1)   # (1, 3, H, W)
    y_rot = rotate_output_tensor(y, 90.0)
    assert torch.allclose(y_rot[:, 0], torch.zeros_like(y_rot[:, 0]), atol=1e-5), \
        f"u after 90° CCW should be ~0, got mean={y_rot[:, 0].mean():.4f}"
    assert torch.allclose(y_rot[:, 1], torch.ones_like(y_rot[:, 1]), atol=1e-5), \
        f"v after 90° CCW should be ~1, got mean={y_rot[:, 1].mean():.4f}"


def test_rotate_output_tensor_shape():
    y = torch.randn(3, 3, 64, 64)
    y_rot = rotate_output_tensor(y, 45.0)
    assert y_rot.shape == y.shape


# ══════════════════════════════════════════════════════════════════════════════
# Data & model loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_offcenter_samples(n_samples: int = 32, device: str = "cpu"):
    """Return (x, y) tensors of off-center test samples, both (N, 3, H, W)."""
    data = np.load("data/processed/dataset.npz")
    test_idx = np.load("data/splits/test_idx.npy")

    inputs  = torch.tensor(
        data["inputs"][test_idx], dtype=torch.float32
    ).permute(0, 3, 1, 2)   # (N, 3, H, W)
    targets = torch.tensor(
        data["targets"][test_idx], dtype=torch.float32
    ).permute(0, 3, 1, 2)

    # Filter: geom_mask centroid must deviate > 0.05 from domain centre (0.5, 0.5)
    H, W = inputs.shape[2], inputs.shape[3]
    xs = torch.linspace(0, 1, W)
    ys = torch.linspace(0, 1, H)
    YY, XX = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    masks = inputs[:, 0]                             # (N, H, W)
    mass  = masks.sum(dim=(1, 2)) + 1e-8             # (N,)
    cx    = (masks * XX).sum(dim=(1, 2)) / mass
    cy    = (masks * YY).sum(dim=(1, 2)) / mass
    keep  = ((cx - 0.5).abs() > 0.05) | ((cy - 0.5).abs() > 0.05)
    inputs  = inputs[keep][:n_samples].to(device)
    targets = targets[keep][:n_samples].to(device)
    return inputs, targets


def _load_fno_baseline(device: str = "cpu"):
    from omegaconf import OmegaConf
    from src.models.model_factory import build_model
    cfg = OmegaConf.load("configs/fno_baseline.yaml")
    model = build_model(
        cfg.model_type,
        in_channels=cfg.in_channels, out_channels=cfg.out_channels,
        modes=cfg.fno_modes, width=cfg.fno_width, n_layers=cfg.n_layers,
    )
    ckpt = Path("models/fno_baseline/best.pt")
    assert ckpt.exists(), f"FNO baseline checkpoint missing: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


def _load_equivariant_fno(device: str = "cpu"):
    from omegaconf import OmegaConf
    from src.models.model_factory import build_model
    ckpt = Path("models/equivariant_fno/best.pt")
    if not ckpt.exists():
        return None
    cfg = OmegaConf.load("configs/equivariant_fno.yaml")
    model = build_model(
        cfg.model_type,
        in_channels=cfg.in_channels, out_channels=cfg.out_channels,
        modes=cfg.fno_modes, width=cfg.fno_width, n_layers=cfg.n_layers,
        group=cfg.group,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


# ══════════════════════════════════════════════════════════════════════════════
# Strategy inference functions
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _strategy_naive_fno(model, x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Strategy 1: directly feed rotated input to FNO."""
    return model(rotate_input_tensor(x, angle_deg))


@torch.no_grad()
def _strategy_tta_canonical(model, x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Strategy 2: rotate all channels back by -angle, predict, rotate forward by +angle.

    rotate_output_tensor uses rotate_flow_field so velocity components are
    transformed by the 2-D rotation matrix, not just spatially moved.
    """
    x_canonical = rotate_input_tensor(x, -angle_deg)
    pred_canonical = model(x_canonical)
    return rotate_output_tensor(pred_canonical, angle_deg)


@torch.no_grad()
def _strategy_equivariant(model, x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Strategy 3: equivariant FNO with rotated input (no TTA)."""
    return model(rotate_input_tensor(x, angle_deg))


# ══════════════════════════════════════════════════════════════════════════════
# Tests for the experiment
# ══════════════════════════════════════════════════════════════════════════════

def test_strategy_naive_fno_output_shape():
    fno = _load_fno_baseline()
    x, y = _load_offcenter_samples(n_samples=4)
    pred = _strategy_naive_fno(fno, x, 90.0)
    assert pred.shape == y.shape


def test_strategy_tta_output_shape():
    fno = _load_fno_baseline()
    x, y = _load_offcenter_samples(n_samples=4)
    pred = _strategy_tta_canonical(fno, x, 90.0)
    assert pred.shape == y.shape


def test_tta_outperforms_naive_at_90():
    """TTA (Strategy 2) must significantly outperform naive FNO (Strategy 1) at 90°."""
    from src.training.metrics import relative_l2_error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = _load_offcenter_samples(n_samples=16, device=device)
    fno = _load_fno_baseline(device=device)
    gt = rotate_output_tensor(y, 90.0)
    err_naive = relative_l2_error(_strategy_naive_fno(fno, x, 90.0), gt)
    err_tta   = relative_l2_error(_strategy_tta_canonical(fno, rotate_input_tensor(x, 90.0), 90.0), gt)
    print(f"\n  Naive FNO @ 90°:   {err_naive * 100:.2f}%")
    print(f"  TTA Canon @ 90°:   {err_tta   * 100:.2f}%")
    assert err_tta < err_naive, (
        f"TTA ({err_tta*100:.2f}%) should beat naive ({err_naive*100:.2f}%)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Full experiment runner
# ══════════════════════════════════════════════════════════════════════════════

_ANGLES = [0, 90, 180, 270]
_STRATEGY_NAMES = {
    1: "Naive FNO",
    2: "TTA Canonicalized",
    3: "Equivariant FNO",
}


def _run_experiment(device: str = "cpu"):
    from src.training.metrics import relative_l2_error
    x, y = _load_offcenter_samples(n_samples=32, device=device)
    fno   = _load_fno_baseline(device=device)
    equi  = _load_equivariant_fno(device=device)

    results = {1: {}, 2: {}, 3: {}}
    for angle in _ANGLES:
        gt = rotate_output_tensor(y, angle)
        x_rot = rotate_input_tensor(x, angle)
        results[1][angle] = relative_l2_error(_strategy_naive_fno(fno, x, angle), gt)
        results[2][angle] = relative_l2_error(_strategy_tta_canonical(fno, x_rot, angle), gt)
        if equi is not None:
            results[3][angle] = relative_l2_error(_strategy_equivariant(equi, x, angle), gt)
        else:
            results[3][angle] = None
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Standalone execution: full experiment + plot
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running equivariance experiment on {device} …")

    results = _run_experiment(device=device)

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 62)
    header = f"{'Angle':>6}  {'Naive FNO':>13}  {'TTA Canon':>13}  {'EquiFNO':>13}"
    print(header)
    print("-" * 62)
    for angle in _ANGLES:
        r1 = results[1][angle]
        r2 = results[2][angle]
        r3 = results[3][angle]
        s3 = f"{r3 * 100:>12.2f}%" if r3 is not None else "         N/A"
        print(f"{angle:>5}°  {r1 * 100:>12.2f}%  {r2 * 100:>12.2f}%  {s3}")

    # ── Bar chart ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(_ANGLES))
    w = 0.25

    vals1 = [results[1][a] * 100 for a in _ANGLES]
    vals2 = [results[2][a] * 100 for a in _ANGLES]
    vals3 = [results[3][a] * 100 if results[3][a] is not None else 0.0 for a in _ANGLES]

    ax.bar(x_pos - w, vals1, w, label="Strategy 1: Naive FNO",        color="steelblue")
    ax.bar(x_pos,     vals2, w, label="Strategy 2: TTA Canonicalized", color="darkorange")
    if any(results[3][a] is not None for a in _ANGLES):
        ax.bar(x_pos + w, vals3, w, label="Strategy 3: Equivariant FNO", color="forestgreen")

    ax.set_xlabel("Rotation Angle", fontsize=12)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=12)
    ax.set_title("Equivariance Generalization Comparison — ENM 5320 Milestone 3", fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{a}°" for a in _ANGLES])
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "equivariance_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
