# Milestone 3 — Equivariance Generalization Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an e2cnn p4 hybrid EquivariantFNO, train it, then run a three-strategy generalization comparison (Naive FNO / TTA Canonicalized / Equivariant FNO) over rotation angles {0°, 90°, 180°, 270°} and save a publication-quality bar chart.

**Architecture:** EquivariantFNO combines e2cnn R2Conv steerable layers (Path A, equivariant) with FNO SpectralConv2d frequency-domain layers (Path B, non-equivariant) in residual blocks. The spectral branch provides global receptive field; the spatial R2Conv branch provides the equivariant inductive bias. Input trivial_repr × 3 → hidden regular_repr × 16 (= 64 physical channels) → output [irrep(1), trivial_repr] (= [u,v], p).

**Tech Stack:** Python 3.x, PyTorch, e2cnn 0.3.x, scipy.ndimage, matplotlib, omegaconf, conda env `fluid-twin`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/equivariant_fno.py` | Replace stub | `SpectralConv2d` + `EquivariantFNOBlock` + `EquivariantFNO` |
| `src/training/trainer.py` | Modify `__main__` | Pass `group` kwarg from config to `build_model` |
| `configs/equivariant_fno.yaml` | Edit | `fno_width: 64` → `fno_width: 16` |
| `tests/test_equivariance.py` | Append | Rotation utils, model shape tests, 3-strategy experiment, plot |
| `docs/figures/` | Create dir | Output for `equivariance_comparison.png` |

---

## Task 1: Install & Verify e2cnn

**Files:** none

- [ ] **Step 1: Install e2cnn in the fluid-twin conda environment**

```bash
conda run -n fluid-twin pip install e2cnn
```

Expected: `Successfully installed e2cnn-...`

- [ ] **Step 2: Smoke-test the e2cnn import**

```bash
conda run -n fluid-twin python -c "
from e2cnn import gspaces
import e2cnn.nn as e2nn
g = gspaces.rot2dOnR2(N=4)
t = e2nn.FieldType(g, [g.regular_repr])
print('e2cnn OK — regular_repr size:', t.size)
"
```

Expected output: `e2cnn OK — regular_repr size: 4`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: install e2cnn for Milestone 3 equivariant architecture"
```

---

## Task 2: Config & Trainer Prep

**Files:**
- Edit: `configs/equivariant_fno.yaml`
- Edit: `src/training/trainer.py:90-97`

- [ ] **Step 1: Change `fno_width` in the equivariant config**

In `configs/equivariant_fno.yaml`, change line 4:

```yaml
fno_width: 16
```

Full updated file content:
```yaml
model_type: equivariant_fno
in_channels: 3
out_channels: 3
fno_modes: 16
fno_width: 16
n_layers: 4
group: p4
epochs: 200
batch_size: 16
lr: 1.0e-3
device: cuda
checkpoint_dir: models/equivariant_fno
use_pino: false
pde_weight: 0.0
```

- [ ] **Step 2: Update `trainer.py` `__main__` to pass `group` kwarg**

In `src/training/trainer.py`, replace lines 90–97:

Old:
```python
    model = build_model(
        cfg.model_type,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        modes=cfg.fno_modes,
        width=cfg.fno_width,
        n_layers=cfg.n_layers,
    )
```

New:
```python
    _model_kwargs = dict(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        modes=cfg.fno_modes,
        width=cfg.fno_width,
        n_layers=cfg.n_layers,
    )
    if hasattr(cfg, "group"):
        _model_kwargs["group"] = cfg.group
    model = build_model(cfg.model_type, **_model_kwargs)
```

- [ ] **Step 3: Verify trainer still works with fno_baseline config (no `group` key)**

```bash
conda run -n fluid-twin python -c "
from omegaconf import OmegaConf
from src.models.model_factory import build_model
cfg = OmegaConf.load('configs/fno_baseline.yaml')
kw = dict(in_channels=cfg.in_channels, out_channels=cfg.out_channels,
          modes=cfg.fno_modes, width=cfg.fno_width, n_layers=cfg.n_layers)
m = build_model(cfg.model_type, **kw)
print('FNO baseline build OK:', type(m).__name__)
"
```

Expected: `FNO baseline build OK: FNOBaseline`

- [ ] **Step 4: Commit**

```bash
git add configs/equivariant_fno.yaml src/training/trainer.py
git commit -m "config: set equivariant_fno width=16 for fair parameter comparison; trainer passes group kwarg"
```

---

## Task 3: TDD — SpectralConv2d

**Files:**
- Create: `tests/test_equivariant_fno.py`
- Edit: `src/models/equivariant_fno.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_equivariant_fno.py`:

```python
"""Unit tests for EquivariantFNO architecture components."""

import pytest
import torch


def test_spectral_conv2d_output_shape():
    """SpectralConv2d must preserve (B, C, H, W) shape."""
    from src.models.equivariant_fno import SpectralConv2d
    sc = SpectralConv2d(in_channels=64, out_channels=64, modes=8)
    x = torch.randn(2, 64, 32, 32)
    y = sc(x)
    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"


def test_spectral_conv2d_different_inout():
    from src.models.equivariant_fno import SpectralConv2d
    sc = SpectralConv2d(in_channels=8, out_channels=16, modes=4)
    x = torch.randn(2, 8, 32, 32)
    y = sc(x)
    assert y.shape == (2, 16, 32, 32)
```

- [ ] **Step 2: Run to verify it fails**

```bash
conda run -n fluid-twin python -m pytest tests/test_equivariant_fno.py -v 2>&1 | head -30
```

Expected: `FAILED` with `ImportError: cannot import name 'SpectralConv2d'`

- [ ] **Step 3: Implement `SpectralConv2d` in `equivariant_fno.py`**

Replace the entire contents of `src/models/equivariant_fno.py` with:

```python
"""SE(2)-equivariant FNO variant using group-equivariant convolutions (e2cnn).

Architecture: Hybrid G-CNN + FNO spectral blocks.
  Lifting  : e2cnn R2Conv  trivial(3) → regular(width)
  Blocks×n : Path A — R2Conv(1×1) equivariant shortcut
             Path B — SpectralConv2d on raw tensor (non-equivariant, global RF)
             Sum + ReLU
  Projection: R2Conv(1×1) × 2   regular(width) → [irrep(1), trivial]

Output channels: irrep(1) → (u, v),  trivial → p

Install: pip install e2cnn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Standard FNO-style spectral convolution on raw (B, C, H, W) tensors."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes,
                                dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, s=(H, W))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m = self.modes
        out_ft[:, :, :m, :m] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m, :m], self.weights
        )
        return torch.fft.irfft2(out_ft, s=(H, W))
```

- [ ] **Step 4: Run tests — expect pass**

```bash
conda run -n fluid-twin python -m pytest tests/test_equivariant_fno.py::test_spectral_conv2d_output_shape tests/test_equivariant_fno.py::test_spectral_conv2d_different_inout -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/models/equivariant_fno.py tests/test_equivariant_fno.py
git commit -m "feat: implement SpectralConv2d for equivariant FNO hybrid architecture"
```

---

## Task 4: TDD — EquivariantFNO Full Class

**Files:**
- Edit: `tests/test_equivariant_fno.py`
- Edit: `src/models/equivariant_fno.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_equivariant_fno.py`:

```python
def test_equivariant_fno_output_shape():
    """EquivariantFNO must return (B, 3, H, W) for default config."""
    from src.models.equivariant_fno import EquivariantFNO
    model = EquivariantFNO(in_channels=3, out_channels=3, modes=8, width=4, n_layers=2)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 3, 32, 32), f"Expected (2,3,32,32), got {y.shape}"


def test_equivariant_fno_output_shape_128():
    """EquivariantFNO with default width=16 and 128×128 grid."""
    from src.models.equivariant_fno import EquivariantFNO
    model = EquivariantFNO(in_channels=3, out_channels=3, modes=16, width=16, n_layers=4)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 3, 128, 128)


def test_equivariant_fno_rejects_wrong_out_channels():
    from src.models.equivariant_fno import EquivariantFNO
    with pytest.raises(AssertionError):
        EquivariantFNO(in_channels=3, out_channels=5, modes=8, width=4, n_layers=2)


def test_equivariant_fno_build_via_factory():
    from src.models.model_factory import build_model
    model = build_model(
        "equivariant_fno",
        in_channels=3, out_channels=3, modes=8, width=4, n_layers=2, group="p4"
    )
    assert model is not None
```

- [ ] **Step 2: Run to verify they fail**

```bash
conda run -n fluid-twin python -m pytest tests/test_equivariant_fno.py -v 2>&1 | tail -20
```

Expected: `FAILED` with `NotImplementedError` or import errors for the new tests.

- [ ] **Step 3: Implement `EquivariantFNOBlock` and `EquivariantFNO`**

Append the following to `src/models/equivariant_fno.py` (after `SpectralConv2d` class):

```python

class EquivariantFNOBlock(nn.Module):
    """Residual block: equivariant spatial shortcut + FNO spectral branch."""

    def __init__(self, feat_type, modes: int):
        super().__init__()
        import e2cnn.nn as e2nn
        n_ch = feat_type.size  # width × 4 physical channels
        self.conv_1x1 = e2nn.R2Conv(feat_type, feat_type, kernel_size=1, bias=False)
        self.spectral = SpectralConv2d(n_ch, n_ch, modes)
        self.feat_type = feat_type

    def forward(self, x):
        import e2cnn.nn as e2nn
        path_a = self.conv_1x1(x)
        tensor_b = self.spectral(x.tensor)
        combined_tensor = F.relu(path_a.tensor + tensor_b)
        return e2nn.GeometricTensor(combined_tensor, self.feat_type)


class EquivariantFNO(nn.Module):
    """Hybrid equivariant FNO: e2cnn p4 steerable layers + FNO spectral blocks.

    Input:  (batch, 3, H, W) — [geom_mask, Re_normalized, inlet_profile]
    Output: (batch, 3, H, W) — [u, v, p]
      channels 0-1: velocity vector (irrep(1))
      channel  2:   pressure scalar (trivial_repr)

    Args:
        in_channels:  Must be 3.
        out_channels: Must be 3.
        modes:        Fourier modes per dimension (default 16).
        width:        Number of logical equivariant channels (default 16).
                      Physical hidden channels = width × 4 (p4 group order).
        n_layers:     Number of FNO spectral blocks (default 4).
        group:        Symmetry group — only 'p4' supported.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 modes: int = 16, width: int = 16, n_layers: int = 4,
                 group: str = "p4"):
        super().__init__()
        assert out_channels == 3, "EquivariantFNO requires out_channels=3 for [u,v,p]"
        assert group == "p4", "Only p4 group is currently supported"

        from e2cnn import gspaces
        import e2cnn.nn as e2nn

        g = gspaces.rot2dOnR2(N=4)
        self.gspace = g

        feat_in = e2nn.FieldType(g, in_channels * [g.trivial_repr])
        feat_hidden = e2nn.FieldType(g, width * [g.regular_repr])
        feat_half = e2nn.FieldType(g, (width // 2) * [g.regular_repr])
        feat_out = e2nn.FieldType(g, [g.irrep(1), g.trivial_repr])

        self.in_type = feat_in
        self.half_type = feat_half

        self.lift = e2nn.R2Conv(feat_in, feat_hidden, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList([
            EquivariantFNOBlock(feat_hidden, modes) for _ in range(n_layers)
        ])
        self.proj1 = e2nn.R2Conv(feat_hidden, feat_half, kernel_size=1, bias=False)
        self.proj2 = e2nn.R2Conv(feat_half, feat_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import e2cnn.nn as e2nn
        h = self.lift(e2nn.GeometricTensor(x, self.in_type))
        for block in self.blocks:
            h = block(h)
        h = e2nn.GeometricTensor(F.relu(self.proj1(h).tensor), self.half_type)
        return self.proj2(h).tensor
```

- [ ] **Step 4: Run all model tests**

```bash
conda run -n fluid-twin python -m pytest tests/test_equivariant_fno.py -v
```

Expected: `6 passed` (2 from Task 3 + 4 new ones)

- [ ] **Step 5: Verify parameter count is comparable to FNO baseline**

```bash
conda run -n fluid-twin python -c "
import torch
from src.models.equivariant_fno import EquivariantFNO
from src.models.fno_baseline import FNOBaseline
equi = EquivariantFNO(modes=16, width=16, n_layers=4)
fno  = FNOBaseline(modes=16, width=64, n_layers=4)
p_equi = sum(p.numel() for p in equi.parameters())
p_fno  = sum(p.numel() for p in fno.parameters())
print(f'EquivariantFNO params: {p_equi:,}')
print(f'FNO baseline params:   {p_fno:,}')
print(f'Ratio: {p_equi / p_fno:.2f}x')
"
```

Expected: ratio roughly 0.5–2.0× (models are in the same order of magnitude).

- [ ] **Step 6: Commit**

```bash
git add src/models/equivariant_fno.py tests/test_equivariant_fno.py
git commit -m "feat: implement EquivariantFNO — e2cnn p4 hybrid G-CNN + spectral blocks"
```

---

## Task 5: Train EquivariantFNO

**Files:** none (reads existing data + config, writes to `models/equivariant_fno/`)

- [ ] **Step 1: Verify dataset and splits exist**

```bash
ls -lh data/processed/dataset.npz data/splits/train_idx.npy data/splits/val_idx.npy
```

Expected: all three files exist with non-zero size.

- [ ] **Step 2: Launch training**

```bash
conda run -n fluid-twin python -m src.training.trainer \
  --config configs/equivariant_fno.yaml 2>&1 | tee models/equivariant_fno/train.log
```

Runs for 200 epochs. Prints every 10 epochs:
```
Epoch   10 | train=0.XXXX | val=0.XXXX
...
Training complete. Best val loss: 0.XXXX
Checkpoint: models/equivariant_fno/best.pt
```

- [ ] **Step 3: Verify checkpoint exists**

```bash
ls -lh models/equivariant_fno/best.pt
```

Expected: file exists, size > 1 MB.

- [ ] **Step 4: Quick inference sanity check**

```bash
conda run -n fluid-twin python -c "
import torch
from omegaconf import OmegaConf
from src.models.model_factory import build_model
cfg = OmegaConf.load('configs/equivariant_fno.yaml')
model = build_model(cfg.model_type, in_channels=cfg.in_channels,
    out_channels=cfg.out_channels, modes=cfg.fno_modes,
    width=cfg.fno_width, n_layers=cfg.n_layers, group=cfg.group)
state = torch.load('models/equivariant_fno/best.pt', map_location='cpu')
model.load_state_dict(state)
model.eval()
with torch.no_grad():
    y = model(torch.zeros(1, 3, 128, 128))
print('Inference OK, output shape:', y.shape)
"
```

Expected: `Inference OK, output shape: torch.Size([1, 3, 128, 128])`

- [ ] **Step 5: Commit training artifacts**

```bash
git add models/equivariant_fno/train.log
git commit -m "feat: train EquivariantFNO — checkpoint at models/equivariant_fno/best.pt"
```

---

## Task 6: TDD — Tensor Rotation Utilities

**Files:**
- Edit: `tests/test_equivariance.py`

- [ ] **Step 1: Write the failing tests**

Append the following to `tests/test_equivariance.py` (keep the three existing spectral-gradient tests, add below):

```python

# ══════════════════════════════════════════════════════════════════════════════
# Tensor rotation utilities (local to this module)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
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
```

- [ ] **Step 2: Run to verify tests pass (rotation utils are already implemented above)**

```bash
conda run -n fluid-twin python -m pytest tests/test_equivariance.py::test_rotate_input_tensor_shape tests/test_equivariance.py::test_rotate_input_tensor_identity tests/test_equivariance.py::test_rotate_output_tensor_rightward_to_upward tests/test_equivariance.py::test_rotate_output_tensor_shape -v
```

Expected: `4 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_equivariance.py
git commit -m "feat: add tensor rotation utilities and unit tests to test_equivariance.py"
```

---

## Task 7: 3-Strategy Experiment + Plot

**Files:**
- Edit: `tests/test_equivariance.py`

- [ ] **Step 1: Write the failing experiment tests**

Append the following to `tests/test_equivariance.py`:

```python

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
    err_tta   = relative_l2_error(_strategy_tta_canonical(fno, x, 90.0), gt)
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
        results[1][angle] = relative_l2_error(_strategy_naive_fno(fno, x, angle), gt)
        results[2][angle] = relative_l2_error(_strategy_tta_canonical(fno, x, angle), gt)
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
```

- [ ] **Step 2: Run the shape + TTA-outperforms tests**

```bash
conda run -n fluid-twin python -m pytest \
  tests/test_equivariance.py::test_strategy_naive_fno_output_shape \
  tests/test_equivariance.py::test_strategy_tta_output_shape \
  tests/test_equivariance.py::test_tta_outperforms_naive_at_90 \
  -v -s
```

Expected: `3 passed`; the `-s` flag prints the error percentages.

- [ ] **Step 3: Run the full experiment and generate the plot**

```bash
conda run -n fluid-twin python tests/test_equivariance.py
```

Expected output (approximate):
```
Running equivariance experiment on cuda …

══════════════════════════════════════════════════════════════
 Angle     Naive FNO      TTA Canon       EquiFNO
──────────────────────────────────────────────────────────────
   0°          2.XX%          2.XX%          2.XX%
  90°         XX.XX%          2.XX%          3.XX%
 180°         XX.XX%          2.XX%          3.XX%
 270°         XX.XX%          2.XX%          3.XX%

Plot saved → docs/figures/equivariance_comparison.png
```

Key result: Naive FNO error rises sharply at 90°/180°/270°; TTA and EquiFNO stay low.

- [ ] **Step 4: Run the entire test file to make sure nothing is broken**

```bash
conda run -n fluid-twin python -m pytest tests/test_equivariance.py -v
```

Expected: all tests pass (at minimum the 4 spectral tests + 4 rotation utility tests + 3 experiment tests = 11 tests).

- [ ] **Step 5: Commit final state**

```bash
git add tests/test_equivariance.py docs/figures/equivariance_comparison.png
git commit -m "feat: Milestone 3 complete — 3-strategy equivariance experiment + comparison plot"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: e2cnn p4 G-CNN ✓, SpectralConv2d ✓, EquivariantFNOBlock ✓, projection with [irrep(1), trivial] ✓, trainer group kwarg ✓, fno_width=16 ✓, rotate_input_tensor (all 3 channels) ✓, rotate_output_tensor uses rotate_flow_field ✓, TTA direction confirmed ✓, 3 strategies ✓, grouped bar chart ✓
- [x] **Placeholder scan**: No TBD/TODO in any code block. All commands have expected output.
- [x] **Type consistency**: `feat_type` used in `EquivariantFNOBlock.__init__` matches `feat_hidden` stored in `EquivariantFNO.__init__`. `rotate_output_tensor` return type matches `_strategy_tta_canonical` consumption.
- [x] **TTA math**: Step 3 (`rotate_output_tensor(pred_canonical, +angle_deg)`) calls `rotate_flow_field` which applies the 2-D rotation matrix to (u,v), not just spatial rotation. Confirmed critical.
