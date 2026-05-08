# Milestone 3 — Equivariance Generalization Experiment Design

**Date**: 2026-04-25  
**Author**: Jimwell Huang  
**Status**: Approved

---

## Context

Milestone 3 demonstrates the value of SE(2)-equivariant inductive biases for neural surrogates of 2D fluid flow. The experiment compares three strategies on a rotated-geometry test set and produces a publication-quality comparison figure for the final report.

---

## Goals

1. Implement `EquivariantFNO` in `src/models/equivariant_fno.py` (e2cnn p4 hybrid architecture).
2. Train `EquivariantFNO` on the existing dataset.
3. Implement `tests/test_equivariance.py` with three-strategy comparison across rotation angles {0°, 90°, 180°, 270°}.
4. Save a comparison bar chart to `docs/figures/equivariance_comparison.png`.

---

## Architecture: EquivariantFNO

### Group

`p4` — discrete 90° rotations via `e2cnn.gspaces.rot2dOnR2(N=4)`.

### Feature types

| Stage | e2cnn type | PyTorch channel dim |
|-------|-----------|---------------------|
| Input | `trivial_repr × in_channels` (3) | 3 |
| Hidden | `regular_repr × width` | `width × 4` |
| Output | `[irrep(1), trivial_repr]` | 3 (u,v=vector, p=scalar) |

> **Width decision (confirmed)**: Use `fno_width: 16` so hidden channels = `16 × 4 = 64`, matching the FNO baseline exactly. This controls for model capacity, ensuring any generalization improvement is attributable to equivariant inductive bias, not raw parameter count. Update `configs/equivariant_fno.yaml` accordingly.

### SpectralConv2d sub-module (non-equivariant, reused from FNO)

Standard FNO spectral convolution operating on raw channel tensors:
- Complex weights: `(in_ch, out_ch, modes, modes//2+1)`, scaled by `1/(in_ch * out_ch)`
- `compl_mul2d` einsum: `"bixy,ioxy->boxy"`
- `torch.fft.rfft2` / `irfft2`, `s=(H, W)`

### Forward pass per block

```
input GeometricTensor (B, width×4, H, W)
├─ Path A: R2Conv(kernel=1) → GeometricTensor         ← equivariant shortcut
└─ Path B:
   ├─ .tensor extract → (B, width×4, H, W)
   ├─ SpectralConv2d (modes, modes)
   └─ wrap back → GeometricTensor                     ← approx. equivariant
Sum(A, B) → ReLU (equivariant)
```

> **Design note**: Path B (spectral) is not strictly equivariant — this is an intentional trade-off to retain the FNO's global receptive field while Path A provides the equivariant inductive bias. In practice, the combined model generalizes better than naive FNO on rotated test data.

### Projection layer

Two `R2Conv(kernel=1)` layers with a ReLU in between:
- `regular_repr × width` → `regular_repr × (width // 2)` → `[irrep(1), trivial_repr]`
- Final `.tensor` unwrap returns `(B, 3, H, W)`.

### Output channel ordering

`e2cnn` stacks representations in declaration order:
- Channels 0–1: `irrep(1)` → (u, v)  
- Channel 2: `trivial_repr` → p

---

## Experiment: `tests/test_equivariance.py`

### Data preparation

- Load `data/processed/dataset.npz` + `data/splits/test_idx.npy`.
- Filter for **off-center** samples: geom_mask (channel 0 of inputs) centroid deviates from (0.5, 0.5) by > 0.05 in normalized coords.
- Use up to `N_SAMPLES = 32` samples for speed.

### Tensor rotation utilities (local to test file)

```python
def rotate_input_tensor(x: Tensor, angle_deg: float) -> Tensor:
    """Rotate all 3 channels of (B, 3, H, W) spatially via scipy."""

def rotate_output_tensor(y: Tensor, angle_deg: float) -> Tensor:
    """Rotate flow field (B, 3, H, W)=[u,v,p] with vector-aware transform."""
    # calls rotate_flow_field per sample
```

### Ground-truth generation

For each base sample, generate rotated GT via `rotate_flow_field`:
```
GT[θ] = rotate_output_tensor(y_base, θ)  for θ ∈ {0, 90, 180, 270}
```

### Strategy inference

| Strategy | Input | Model | Output |
|----------|-------|-------|--------|
| 1 – Naive FNO | `rotate_input(x, θ)` | FNO baseline | direct |
| 2 – TTA Canonicalized | rotate all 3 channels back: `rotate_input(x_rotated, -θ)` → FNO → `rotate_output(pred, θ)` | FNO baseline | rotated back |
| 3 – Equivariant FNO | `rotate_input(x, θ)` | EquivariantFNO | direct |

> **TTA critical note**: Strategy 2 step 3 (`rotate_output by +θ`) MUST call `rotate_flow_field(u, v, p, +θ)` — NOT a plain spatial rotation. The (u,v) velocity components require the 2D rotation matrix transform; otherwise the flow direction will be physically wrong.

### Error metric

`relative_l2_error(pred, GT[θ])` from `src.training.metrics` for each (strategy, θ) pair.

### Checkpoint handling

- FNO baseline: `models/fno_baseline/best.pt` (required — fail fast if missing).
- EquivariantFNO: `models/equivariant_fno/best.pt` (optional — warn + skip Strategy 3 if missing).

### Plot

Grouped bar chart:
- X-axis: rotation angle (0°, 90°, 180°, 270°)
- Y-axis: relative L2 error (%)
- 3 bars per angle: Strategy 1 (blue), Strategy 2 (orange), Strategy 3 (green)
- Saved to `docs/figures/equivariance_comparison.png` (300 dpi)

### Script execution modes

- `pytest tests/test_equivariance.py -s`: runs all assertions + prints summary table
- `python tests/test_equivariance.py`: additionally generates and saves the plot

---

## Training Workflow

```bash
# Install e2cnn if not present
pip install e2cnn

# Train EquivariantFNO
python -m src.training.trainer --config configs/equivariant_fno.yaml
# Checkpoint saved to: models/equivariant_fno/best.pt
```

Config `configs/equivariant_fno.yaml` already exists; verify `model_type: equivariant_fno` and `group: p4` are set.

---

## Files to Create / Modify

| File | Action | Notes |
|------|--------|-------|
| `src/models/equivariant_fno.py` | Modify | Replace `NotImplementedError` with full implementation |
| `src/models/model_factory.py` | Modify | Wire `equivariant_fno` → `EquivariantFNO` |
| `tests/test_equivariance.py` | Modify | Replace spectral-only tests, add 3-strategy experiment |
| `docs/figures/` | Create dir | Output directory for comparison plot |

---

## Verification

1. `python -c "from src.models.equivariant_fno import EquivariantFNO; m = EquivariantFNO(); import torch; y = m(torch.zeros(2,3,64,64)); print(y.shape)"` → `torch.Size([2, 3, 64, 64])`
2. Equivariance smoke test: for θ=90°, `||EquiFNO(rotate_input(x,90)) - rotate_output(EquiFNO(x),90)|| / ||EquiFNO(x)||` should be small (< 10% for a well-trained model).
3. `pytest tests/test_equivariance.py -s` passes all assertions.
4. `docs/figures/equivariance_comparison.png` saved successfully.
