# src/inference/server.py
"""FastAPI inference server: (cx, cy, re) → (u, v, p) fields + USD scene path.

Accessible from Windows Omniverse Kit at http://localhost:8000.

Usage:
    python -m src.inference.server \
        --checkpoint models/fno_baseline/best.pt \
        --model-type fno_baseline
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fluid Digital Twin Inference Server")

# ── Global state ──────────────────────────────────────────────────────────────

_model = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_stats: dict | None = None   # {u_mean, u_std, v_mean, v_std, p_mean, p_std}
_geometry = None              # CylinderGeometry instance
_v_max: float = 2.0          # computed from stats at startup; passed to USD exporter
_nx: int = 128
_ny: int = 128

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class FlowRequest(BaseModel):
    cx: float           # cylinder center x, normalized [0, 1]
    cy: float           # cylinder center y, normalized [0, 1]
    re: float           # Reynolds number, physical units [20, 200]
    export_usd: bool = True


class FlowResponse(BaseModel):
    u: list[list[float]]         # (H, W) denormalized x-velocity
    v: list[list[float]]         # (H, W) denormalized y-velocity
    p: list[list[float]]         # (H, W) denormalized pressure
    usd_path: str | None = None  # Windows-format path for Omniverse hot-reload


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_input(cx: float, cy: float, re: float) -> torch.Tensor:
    """Build (1, 3, H, W) input tensor from cylinder parameters.

    Channel layout matches the training preprocessor:
        0 — geometry mask (1.0 inside cylinder, 0.0 fluid)
        1 — Re normalized to [0, 1]: (re - 20) / 180
        2 — inlet profile: 1.0 at left column (j=0), 0.0 elsewhere
    """
    mask = _geometry.make_mask(cx, cy).astype(np.float32)
    re_norm = (re - 20.0) / 180.0
    inlet = np.zeros((_ny, _nx), dtype=np.float32)
    inlet[:, 0] = 1.0
    x = np.stack([mask, np.full_like(mask, re_norm), inlet], axis=0)  # (3, H, W)
    return torch.from_numpy(x).unsqueeze(0)  # (1, 3, H, W)


def _denormalize(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reverse Z-score normalization using training-set statistics."""
    u = pred[0] * _stats["u_std"] + _stats["u_mean"]
    v = pred[1] * _stats["v_std"] + _stats["v_mean"]
    p = pred[2] * _stats["p_std"] + _stats["p_mean"]
    return u, v, p


# ── Startup ───────────────────────────────────────────────────────────────────


def load_model(
    checkpoint_path: str,
    model_type: str = "fno_baseline",
    stats_path: str = "data/processed/stats.npz",
) -> None:
    """Load model checkpoint, normalization stats, and geometry into global state."""
    global _model, _stats, _geometry, _v_max

    from src.models import build_model
    from src.data_gen.geometry import CylinderGeometry

    _model = build_model(model_type)
    _model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    _model.eval().to(_device)

    raw = np.load(stats_path)
    _stats = {k: float(raw[k]) for k in ["u_mean", "u_std", "v_mean", "v_std", "p_mean", "p_std"]}

    # Stable color-scale ceiling: 99.7th-percentile velocity magnitude from
    # training stats, so USD colors don't flicker as the cylinder moves.
    u_max = _stats["u_mean"] + 3.0 * _stats["u_std"]
    v_max_est = 3.0 * _stats["v_std"]
    _v_max = float(np.sqrt(u_max ** 2 + v_max_est ** 2))

    _geometry = CylinderGeometry(nx=_nx, ny=_ny)


# ── Endpoint ──────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=FlowResponse)
async def predict(request: FlowRequest) -> FlowResponse:
    """Run neural operator inference for a single (geometry, Re) query."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Start server with --checkpoint.",
        )

    x = _build_input(request.cx, request.cy, request.re).to(_device)
    with torch.no_grad():
        pred = _model(x)[0].cpu().numpy()  # (3, H, W)

    u, v, p = _denormalize(pred)

    usd_path = None
    if request.export_usd:
        from src.inference.usd_exporter import export_flow_to_usd
        usd_path = export_flow_to_usd(u, v, p, v_max=_v_max)

    return FlowResponse(u=u.tolist(), v=v.tolist(), p=p.tolist(), usd_path=usd_path)


# ── CLI entry point ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Fluid Digital Twin Inference Server")
    parser.add_argument("--checkpoint", default="models/fno_baseline/best.pt",
                        help="Path to model .pt checkpoint")
    parser.add_argument("--model-type", default="fno_baseline",
                        choices=["fno_baseline", "equivariant_fno"],
                        help="Model architecture to load")
    parser.add_argument("--stats", default="data/processed/stats.npz",
                        help="Path to normalization stats .npz")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.checkpoint, args.model_type, args.stats)
    uvicorn.run(app, host=args.host, port=args.port)
