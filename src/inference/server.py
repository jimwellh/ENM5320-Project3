"""FastAPI inference server: receives (geometry, Re) → returns (u, v, p) fields.

Accessible from Windows Omniverse Kit at http://localhost:8000.

Usage:
    python -m src.inference.server --checkpoint models/equivariant_fno/best.pt
"""

from __future__ import annotations

import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fluid Digital Twin Inference Server")


class FlowRequest(BaseModel):
    """Inference request payload."""
    geom_mask: list[list[float]]  # (H, W) flattened
    re: float
    nx: int = 128
    ny: int = 128


class FlowResponse(BaseModel):
    """Inference response payload."""
    u: list[list[float]]
    v: list[list[float]]
    p: list[list[float]]


_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str) -> None:
    """Load model checkpoint into global state."""
    global _model
    from src.models import build_model
    _model = build_model("equivariant_fno")
    _model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    _model.eval().to(_device)


@app.post("/predict", response_model=FlowResponse)
async def predict(request: FlowRequest) -> FlowResponse:
    """Run neural operator inference for a single (geometry, Re) query."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    geom = np.array(request.geom_mask, dtype=np.float32)
    re_norm = (request.re - 20.0) / 180.0
    inlet = np.ones_like(geom) * re_norm

    x = np.stack([geom, np.full_like(geom, re_norm), inlet], axis=0)
    x_tensor = torch.tensor(x[None], device=_device)

    with torch.no_grad():
        pred = _model(x_tensor)[0].cpu().numpy()

    return FlowResponse(
        u=pred[0].tolist(),
        v=pred[1].tolist(),
        p=pred[2].tolist(),
    )
