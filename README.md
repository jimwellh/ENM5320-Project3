# ENM 5320 Final Project — Equivariant Fluid Digital Twin (WSL2)

Real-time 2D fluid flow prediction using equivariant neural operators (FNO / SE(2)-FNO), trained on Warp-generated Kármán vortex street data and deployed via FastAPI + NVIDIA Omniverse.

**Conda env**: `fluid-twin` | **Platform**: WSL2 (training) + Windows (Omniverse)


## Project Overview

This project trains equivariant neural operators to predict 2D fluid flow fields (Kármán vortex street — flow past a cylinder) and deploys the trained model in NVIDIA Omniverse for real-time interactive visualization.

**Core scientific question**: Do SE(2)-equivariant neural operators generalize better under geometric transformations than standard FNOs, and how does architectural equivariance compare to data augmentation?

**Course connections** (ENM 5320):
- Symmetry groups and their role in physical laws (SE(2) = rotations + translations in 2D)
- Fourier Neural Operators and spectral methods
- Physics-informed loss via Fourier spectral differentiation (consistent with L3 spectral differentiation)

*Note: This specific folder `ENM5320_Project3` is the WSL part of the project. Check [https://github.com/jimwellh/ENM5320-Project3](https://github.com/jimwellh/ENM5320_FinalProject_Omniverse) for the Windows part.*

---

## Environment

**Platform**: WSL2 (Ubuntu) for ML training + Windows for Omniverse visualization  
**Conda env**: `fluid-twin`

```bash
conda activate fluid-twin
```

**Key packages** (already installed in `fluid-twin`):
- `torch` — PyTorch (CUDA)
- `warp-lang` — NVIDIA Warp (CFD solver kernels)
- `nvidia-physicsnemo` — FNO, TFNO, AFNO, and Transolver implementations
- `scikit-fem` — finite element utilities
- `usd-core` — OpenUSD Python API
- `hydra-core` — config management
- `fastapi`, `uvicorn` — inference server
- `NVIDIA NIM` — for production-grade inference microservice deployment

**GPU**: Check with `nvidia-smi` from WSL2.

**Warp kernel cache**: `~/.cache/warp/` — delete if kernels fail to recompile.

---

## Architecture

```
WSL2 (this repo)                          Windows
─────────────────────────────────         ──────────────────────────────
Warp NS Solver → data/raw/               
Dataset Builder → data/processed/        
FNO / Equivariant FNO training           
PINO loss (Fourier derivatives)          
Inference Server (FastAPI :8000)  ←────→ Omniverse Kit App
USD Exporter → usd_exports/      ←────→ /mnt/c/... shared path
```

WSL2 ↔ Windows communication: shared filesystem via `/mnt/c/Users/<user>/` for USD assets; inference server at `localhost:8000` accessible from Windows.

*Note: The Windows-side Omniverse Kit App should ideally utilize the "Digital Twins for Fluid Simulation" blueprint as a starting template for UI and interactive obstacle manipulation.*

---

## Directory Structure

```
ENM5320_Project3/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # Warp solver output (.npz per simulation)
│   ├── processed/    # Normalized tensors (.npz, shape N×H×W×C)
│   └── splits/       # train/val/test indices (.npy)
├── src/
│   ├── data_gen/     # Milestone 1: CFD data generation
│   ├── models/       # Milestone 2 & 3: FNO and equivariant FNO
│   ├── training/     # Milestone 2-4: training loops, losses, metrics
│   ├── pino/         # Milestone 4: Fourier spectral derivatives
│   ├── inference/    # Milestone 5: FastAPI server + USD exporter
│   └── utils/        # Shared utilities
├── configs/          # Hydra YAML configs
├── notebooks/        # Exploratory analysis
├── models/           # Saved checkpoints
│   ├── fno_baseline/
│   ├── equivariant_fno/
│   └── pino/
├── usd_exports/      # OpenUSD scene files for Omniverse
└── tests/
```

---

## Data Format

**Raw** (`data/raw/re{Re}_x{cx}_y{cy}.npz`):
- `u`, `v`, `p`: velocity and pressure fields, shape `(H, W)`
- `Re`: Reynolds number (scalar)
- `cx`, `cy`: cylinder center position (normalized 0-1)

**Processed** (`data/processed/dataset.npz`):
- `inputs`: shape `(N, H, W, 3)` — channels: `[geom_mask, Re_normalized, inlet_profile]`
- `targets`: shape `(N, H, W, 3)` — channels: `[u, v, p]`

**Grid**: H=W=128 (default), uniform Cartesian, domain `[0,1]×[0,1]`.

---


## Key Commands

```bash
# Generate CFD dataset
conda activate fluid-twin
python -m src.data_gen.dataset_builder --config configs/data_gen.yaml

# Train FNO baseline
python -m src.training.trainer --config configs/fno_baseline.yaml

# Train equivariant FNO
python -m src.training.trainer --config configs/equivariant_fno.yaml

# Evaluate and compare
python -m src.training.metrics --model fno_baseline --checkpoint models/fno_baseline/best.pt
python -m src.training.metrics --model equivariant_fno --checkpoint models/equivariant_fno/best.pt

# Start inference server (accessible from Windows at localhost:8000)
python -m src.inference.server --checkpoint models/equivariant_fno/best.pt

# Export USD scene
python -m src.inference.usd_exporter --output usd_exports/flow_scene.usda
```

---

## Equivariance Implementation Notes

Two options for SE(2)-equivariant FNO (choose one):

1. **Group convolution** (`e2cnn` library): Replace standard conv layers with G-conv over p4 group (90° rotations). Simpler but only discrete rotations.
2. **Steerable feature fields** (`e2cnn` with continuous SO(2)): Full continuous rotation equivariance. More expressive but heavier.

Key experiment (Milestone 3):
- Train on upright cylinder data only
- Test on 90°-rotated cylinder data
- Compare: standard FNO vs equivariant FNO vs FNO + data augmentation

---

## Physics-Informed Loss 

Navier-Stokes residual in 2D (incompressible, steady-state approximation):
```
∇·u = 0                         (continuity)
(u·∇)u = -∇p + (1/Re)∇²u       (momentum)
```

Fourier spectral derivatives (consistent with course L3):
```python
# ∂f/∂x in Fourier space: multiply by i·kx, then IFFT
df_dx = torch.fft.ifft2(1j * kx * torch.fft.fft2(f))
```

---
