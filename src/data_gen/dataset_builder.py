"""Batch dataset generation: runs NS solver over a parameter grid and saves results."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from .warp_ns_solver import NavierStokesSolver
from .geometry import CylinderGeometry


@dataclass
class DatasetConfig:
    re_min: float = 20.0
    re_max: float = 200.0
    n_re: int = 40
    cx_range: tuple[float, float] = (0.3, 0.5)
    cy_range: tuple[float, float] = (0.4, 0.6)
    n_positions: int = 36
    nx: int = 128
    ny: int = 128
    cylinder_radius: float = 0.05  # D = 2 * cylinder_radius = 0.10
    output_dir: str = "data/raw"


class DatasetBuilder:
    """Generates a dataset of (geometry, Re_D) → (u, v, p) instantaneous snapshots.

    Args:
        config: Dataset generation configuration.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.geom = CylinderGeometry(config.nx, config.ny)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def generate(self) -> None:
        """Run all simulations and save individual .npz files to output_dir."""
        re_values = np.linspace(self.config.re_min, self.config.re_max, self.config.n_re)
        n_pos = int(self.config.n_positions ** 0.5)
        cx_values = np.linspace(*self.config.cx_range, n_pos)
        cy_values = np.linspace(*self.config.cy_range, n_pos)

        # Allocate solver once — GPU arrays reused across all samples
        cylinder_diameter = 2.0 * self.config.cylinder_radius
        solver = NavierStokesSolver(
            nx=self.config.nx,
            ny=self.config.ny,
            cylinder_diameter=cylinder_diameter,
        )

        count = 0
        total = self.config.n_re * n_pos * n_pos
        for re in re_values:
            for cx in cx_values:
                for cy in cy_values:
                    out_path = (Path(self.config.output_dir)
                                / f"re{re:.1f}_cx{cx:.3f}_cy{cy:.3f}.npz")
                    if out_path.exists():
                        count += 1
                        continue  # resume support: skip completed files
                    mask = self.geom.make_mask(cx, cy, radius=self.config.cylinder_radius)
                    fields = solver.solve(mask, re=re)
                    np.savez(out_path, geom_mask=mask, re=re, cx=cx, cy=cy, **fields)
                    count += 1
                    print(f"[{count}/{total}] Re={re:.0f} cx={cx:.3f} cy={cy:.3f}")

        print(f"Generated {count} samples → {self.config.output_dir}")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate CFD dataset (Milestone 1)")
    parser.add_argument("--config", default="configs/data_gen.yaml")
    parser.add_argument("--n_re", type=int, default=None, help="Override n_re")
    parser.add_argument("--n_positions", type=int, default=None,
                        help="Override n_positions (must be a perfect square)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_re is not None:
        cfg["n_re"] = args.n_re
    if args.n_positions is not None:
        cfg["n_positions"] = args.n_positions

    config = DatasetConfig(**cfg)
    DatasetBuilder(config).generate()
