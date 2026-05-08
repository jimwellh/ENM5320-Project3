"""Generate a CFD dataset point for an ellipse shape at Re=100."""

import numpy as np
from pathlib import Path
from src.data_gen.warp_ns_solver import NavierStokesSolver
from src.data_gen.geometry import CylinderGeometry

def generate():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "re100.0_ellipse.npz"

    nx = 128
    ny = 128
    # Keep the same vertical diameter (D=0.10) for consistent Re
    # This means ry = 0.05, rx = 0.10 (horizontal ellipse)
    rx = 0.10
    ry = 0.05
    cylinder_diameter = 2.0 * ry # reference length is vertical dimension
    
    geom = CylinderGeometry(nx, ny)
    mask = geom.make_ellipse_mask(cx=0.4, cy=0.5, rx=rx, ry=ry)

    solver = NavierStokesSolver(nx=nx, ny=ny, cylinder_diameter=cylinder_diameter)
    
    re = 100.0
    print(f"Running simulation for ellipse at Re={re}...")
    fields = solver.solve(mask, re=re)
    
    np.savez(out_path, geom_mask=mask, re=re, cx=0.4, cy=0.5, **fields)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate()
