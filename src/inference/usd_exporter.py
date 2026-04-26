# src/inference/usd_exporter.py
"""Export flow field prediction to OpenUSD for Omniverse visualization."""

from __future__ import annotations

import numpy as np
from pathlib import Path

DEFAULT_OUTPUT_DIR = "/mnt/c/Users/jimwell/omniverse_assets"

# Physics-based upper bound for velocity magnitude (Re=20-200, U_inf=1.0).
# Used as a stable per-session color scale so colors don't flicker as the
# cylinder moves. Override via the v_max parameter.
GLOBAL_V_MAX: float = 2.0


def export_flow_to_usd(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: str = "flow_scene.usda",
    v_max: float = GLOBAL_V_MAX,
) -> str:
    """Export (u, v, p) fields to an OpenUSD quad mesh for Omniverse.

    Creates a flat grid mesh (H×W vertices, (H-1)×(W-1) quads) with:
    - displayColor primvar: velocity magnitude clipped to [0, v_max] and
      mapped blue→red. v_max is fixed (not per-frame) so colors are stable.
    - custom:u_field, custom:v_field, custom:p_field: raw float arrays for
      Omniverse scripting and hot-reload.

    Args:
        u: x-velocity, shape (H, W) — physical units (denormalized).
        v: y-velocity, shape (H, W) — physical units (denormalized).
        p: pressure, shape (H, W) — physical units (denormalized).
        output_dir: Directory to write the .usda file. Creates if missing.
        filename: Output filename (must end in .usda).
        v_max: Fixed velocity magnitude ceiling for color mapping.
               Default GLOBAL_V_MAX=2.0 covers Re=20-200 training range.

    Returns:
        Absolute path to the written file. Paths under /mnt/c/ are
        returned in Windows format (C:/...) for Omniverse hot-reload.
    """
    try:
        from pxr import Usd, UsdGeom, Sdf, Vt, Gf
    except ImportError:
        raise ImportError("Install usd-core: pip install usd-core")

    output_path = str(Path(output_dir) / filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(output_path)
    stage.SetMetadata("upAxis", "Y")

    UsdGeom.Xform.Define(stage, "/FlowField")
    mesh = UsdGeom.Mesh.Define(stage, "/FlowField/VelocityMesh")

    h, w = u.shape

    # (h × w) vertex grid: x = column / (w-1), z = row / (h-1), y = 0
    points = Vt.Vec3fArray([
        Gf.Vec3f(i / max(w - 1, 1), 0.0, j / max(h - 1, 1))
        for j in range(h)
        for i in range(w)
    ])
    mesh.GetPointsAttr().Set(points)

    # (h-1) × (w-1) quads — CCW winding: BL, BR, TR, TL
    face_vertex_counts = Vt.IntArray([4] * ((h - 1) * (w - 1)))
    face_vertex_indices = Vt.IntArray([
        idx
        for j in range(h - 1)
        for i in range(w - 1)
        for idx in (j * w + i, j * w + i + 1, (j + 1) * w + i + 1, (j + 1) * w + i)
    ])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Velocity magnitude → stable blue-to-red per-vertex color.
    # np.clip ensures values above v_max saturate to red rather than rescaling.
    vel_mag = np.sqrt(u ** 2 + v ** 2)
    mag_norm = np.clip(vel_mag / v_max, 0.0, 1.0).ravel()
    color_pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    color_pv.Set(Vt.Vec3fArray([
        Gf.Vec3f(float(m), 0.0, 1.0 - float(m)) for m in mag_norm
    ]))

    # Raw field data as custom attributes (Omniverse scripting / animation)
    prim = mesh.GetPrim()
    prim.CreateAttribute("custom:u_field", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray(u.ravel().tolist())
    )
    prim.CreateAttribute("custom:v_field", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray(v.ravel().tolist())
    )
    prim.CreateAttribute("custom:p_field", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray(p.ravel().tolist())
    )
    prim.CreateAttribute("custom:field_shape", Sdf.ValueTypeNames.Int2).Set(
        Gf.Vec2i(h, w)
    )

    stage.GetRootLayer().Save()

    # Convert WSL path to Windows format for Omniverse hot-reload
    win_path = output_path.replace("/mnt/c/", "C:/").replace("/mnt/d/", "D:/")
    return win_path
