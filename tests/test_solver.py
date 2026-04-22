"""Tests for the Warp NS solver and geometry utilities."""

import numpy as np
import pytest
from src.data_gen.geometry import CylinderGeometry
from src.data_gen.warp_ns_solver import NavierStokesSolver


def test_cylinder_mask_shape():
    geom = CylinderGeometry(nx=64, ny=64)
    mask = geom.make_mask(cx=0.5, cy=0.5, radius=0.1)
    assert mask.shape == (64, 64)
    assert mask.dtype == bool


def test_cylinder_mask_coverage():
    geom = CylinderGeometry(nx=128, ny=128)
    mask = geom.make_mask(cx=0.5, cy=0.5, radius=0.05)
    assert mask.sum() > 0, "Cylinder mask should contain at least one True pixel"
    assert mask.sum() < 128 * 128, "Cylinder mask should not fill the entire domain"


def test_cylinder_mask_center():
    geom = CylinderGeometry(nx=100, ny=100)
    mask = geom.make_mask(cx=0.5, cy=0.5, radius=0.1)
    # Center pixel should be inside the cylinder
    assert mask[50, 50]


def test_rotate_mask_preserves_shape():
    geom = CylinderGeometry(nx=64, ny=64)
    mask = geom.make_mask(cx=0.5, cy=0.5, radius=0.1)
    rotated = geom.rotate_mask(mask, angle_deg=90.0)
    assert rotated.shape == mask.shape


# ---------------------------------------------------------------------------
# NS Solver physics tests
# ---------------------------------------------------------------------------

def test_solver_output_shape():
    """solve() returns u, v, p each of shape (ny, nx)."""
    solver = NavierStokesSolver(nx=32, ny=32, cylinder_diameter=0.10,
                                dt=0.0002, max_steps=50)
    mask = np.zeros((32, 32), dtype=bool)
    fields = solver.solve(mask, re=100.0)
    for key in ("u", "v", "p"):
        assert key in fields
        assert fields[key].shape == (32, 32), f"{key} shape: {fields[key].shape}"


def test_inlet_bc_enforced():
    """After solve(), inlet column u≈1.0 and v≈0.0."""
    solver = NavierStokesSolver(nx=32, ny=32, cylinder_diameter=0.10,
                                dt=0.0002, max_steps=200)
    mask = np.zeros((32, 32), dtype=bool)
    fields = solver.solve(mask, re=50.0)
    assert np.allclose(fields["u"][:, 0], 1.0, atol=0.05), \
        f"Inlet u max error: {np.abs(fields['u'][:, 0] - 1.0).max():.4f}"
    assert np.allclose(fields["v"][:, 0], 0.0, atol=0.05), \
        f"Inlet v max error: {np.abs(fields['v'][:, 0]).max():.4f}"


def test_cylinder_noslip():
    """Velocity is exactly zero inside the solid cylinder."""
    geom = CylinderGeometry(32, 32)
    mask = geom.make_mask(0.4, 0.5, radius=0.08)
    solver = NavierStokesSolver(nx=32, ny=32, cylinder_diameter=0.16,
                                dt=0.0002, max_steps=200)
    fields = solver.solve(mask, re=100.0)
    assert np.allclose(fields["u"][mask], 0.0, atol=1e-5), \
        f"Cylinder u max: {np.abs(fields['u'][mask]).max():.2e}"
    assert np.allclose(fields["v"][mask], 0.0, atol=1e-5), \
        f"Cylinder v max: {np.abs(fields['v'][mask]).max():.2e}"


def test_mass_conservation_500_jacobi():
    """After 500 Jacobi iters/step, interior |∇·u| mean < 0.01."""
    solver = NavierStokesSolver(nx=32, ny=32, cylinder_diameter=0.10,
                                dt=0.0002, max_steps=300)
    mask = np.zeros((32, 32), dtype=bool)
    fields = solver.solve(mask, re=100.0)
    u, v = fields["u"], fields["v"]
    dx, dy = 1.0 / 31, 1.0 / 31
    du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
    dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    mean_div = np.abs(du_dx + dv_dy).mean()
    assert mean_div < 0.01, f"Mean divergence too high: {mean_div:.5f} (expected < 0.01)"
