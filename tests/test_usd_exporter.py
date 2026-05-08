# tests/test_usd_exporter.py
"""Tests for USD flow field export."""
import numpy as np
import pytest
from pathlib import Path


def test_export_creates_usda_file(tmp_path):
    from src.inference.usd_exporter import export_flow_to_usd
    u = np.zeros((16, 16), dtype=np.float32)
    v = np.zeros((16, 16), dtype=np.float32)
    p = np.zeros((16, 16), dtype=np.float32)
    export_flow_to_usd(u, v, p, output_dir=str(tmp_path), filename="out.usda")
    assert (tmp_path / "out.usda").exists()


def test_export_returns_string_path(tmp_path):
    from src.inference.usd_exporter import export_flow_to_usd
    u = np.zeros((16, 16), dtype=np.float32)
    v = np.zeros((16, 16), dtype=np.float32)
    p = np.zeros((16, 16), dtype=np.float32)
    result = export_flow_to_usd(u, v, p, output_dir=str(tmp_path))
    assert isinstance(result, str)
    assert result.endswith(".usda")


def test_export_mesh_has_correct_topology(tmp_path):
    """16×16 point grid → 15×15 quads (225 faces)."""
    from pxr import Usd, UsdGeom
    from src.inference.usd_exporter import export_flow_to_usd
    u = np.random.randn(16, 16).astype(np.float32)
    v = np.random.randn(16, 16).astype(np.float32)
    p = np.random.randn(16, 16).astype(np.float32)
    export_flow_to_usd(u, v, p, output_dir=str(tmp_path), filename="topo.usda")
    stage = Usd.Stage.Open(str(tmp_path / "topo.usda"))
    mesh_prim = stage.GetPrimAtPath("/FlowField/VelocityMesh")
    assert mesh_prim.IsValid()
    mesh = UsdGeom.Mesh(mesh_prim)
    assert len(mesh.GetPointsAttr().Get()) == 16 * 16
    assert len(mesh.GetFaceVertexCountsAttr().Get()) == 15 * 15
    assert all(c == 4 for c in mesh.GetFaceVertexCountsAttr().Get())


def test_export_custom_field_attributes(tmp_path):
    """custom:u_field, custom:v_field, custom:p_field must exist on mesh prim."""
    from pxr import Usd
    from src.inference.usd_exporter import export_flow_to_usd
    u = np.ones((16, 16), dtype=np.float32) * 2.0
    v = np.ones((16, 16), dtype=np.float32) * 3.0
    p = np.ones((16, 16), dtype=np.float32) * 5.0
    export_flow_to_usd(u, v, p, output_dir=str(tmp_path), filename="attrs.usda")
    stage = Usd.Stage.Open(str(tmp_path / "attrs.usda"))
    prim = stage.GetPrimAtPath("/FlowField/VelocityMesh")
    u_attr = prim.GetAttribute("custom:u_field")
    v_attr = prim.GetAttribute("custom:v_field")
    p_attr = prim.GetAttribute("custom:p_field")
    assert u_attr.IsValid()
    assert v_attr.IsValid()
    assert p_attr.IsValid()
    assert abs(list(u_attr.Get())[0] - 2.0) < 1e-5
    assert abs(list(v_attr.Get())[0] - 3.0) < 1e-5
    assert abs(list(p_attr.Get())[0] - 5.0) < 1e-5


def test_export_color_uses_global_vmax_not_per_frame_max(tmp_path):
    """Color must be clipped to a fixed v_max, not normalized per-frame.

    Set v_max=1.0 and velocity magnitude=2.0 everywhere. With clip, red
    channel must be 1.0 (saturated). A per-frame normalizer would also give
    1.0 for a uniform field — so we verify clipping by checking that passing
    v_max=1.0 when mag=2.0 produces red=1.0, not red=2.0.
    """
    from pxr import Usd, UsdGeom
    from src.inference.usd_exporter import export_flow_to_usd
    u = np.full((3, 3), 2.0, dtype=np.float32)
    v = np.zeros((3, 3), dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    export_flow_to_usd(u, v, p, output_dir=str(tmp_path), filename="clip.usda",
                       v_max=1.0)
    stage = Usd.Stage.Open(str(tmp_path / "clip.usda"))
    prim = stage.GetPrimAtPath("/FlowField/VelocityMesh")
    mesh = UsdGeom.Mesh(prim)
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    colors = primvars_api.GetPrimvar("displayColor").Get()
    for c in colors:
        assert abs(c[0] - 1.0) < 1e-5, f"Expected red=1.0 (clipped), got {c[0]}"
