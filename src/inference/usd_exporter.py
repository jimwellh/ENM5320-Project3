"""Export flow field prediction to OpenUSD for Omniverse visualization.

Writes velocity magnitude as a UsdGeom plane with primvar color data,
or exports to a shared Windows path for Omniverse Kit ingestion.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def export_flow_to_usd(u: np.ndarray, v: np.ndarray, p: np.ndarray,
                       output_path: str = "usd_exports/flow_scene.usda") -> None:
    """Export (u, v, p) fields to an OpenUSD file.

    Args:
        u: Velocity x-component, shape (H, W).
        v: Velocity y-component, shape (H, W).
        p: Pressure field, shape (H, W).
        output_path: Path to write .usda file.
    """
    try:
        from pxr import Usd, UsdGeom, Gf, Vt
    except ImportError:
        raise ImportError("Install usd-core: pip install usd-core")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(output_path)
    stage.SetMetadata("upAxis", "Y")

    xform = UsdGeom.Xform.Define(stage, "/FlowField")
    mesh = UsdGeom.Mesh.Define(stage, "/FlowField/VelocityMesh")

    h, w = u.shape
    velocity_magnitude = np.sqrt(u ** 2 + v ** 2)

    # Build quad mesh for the 2D flow domain
    points = []
    for j in range(h):
        for i in range(w):
            points.append(Gf.Vec3f(i / w, 0.0, j / h))

    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

    # Store velocity magnitude as display color primvar
    primvar_api = UsdGeom.PrimvarsAPI(mesh)
    color_primvar = primvar_api.CreatePrimvar(
        "displayColor", Gf.Vec3f, UsdGeom.Tokens.vertex
    )
    mag_normalized = velocity_magnitude / (velocity_magnitude.max() + 1e-8)
    colors = [Gf.Vec3f(float(v), 0.0, 1.0 - float(v)) for v in mag_normalized.ravel()]
    color_primvar.Set(Vt.Vec3fArray(colors))

    stage.GetRootLayer().Save()
    print(f"USD scene saved → {output_path}")
