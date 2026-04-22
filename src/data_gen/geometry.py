"""Geometry utilities: cylinder mask generation and obstacle parameterization."""

import numpy as np


class CylinderGeometry:
    """Generates solid masks for a circular cylinder on a uniform grid.

    Args:
        nx: Grid resolution in x.
        ny: Grid resolution in y.
    """

    def __init__(self, nx: int = 128, ny: int = 128):
        self.nx = nx
        self.ny = ny
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(xs, ys)

    def make_mask(self, cx: float, cy: float, radius: float = 0.05) -> np.ndarray:
        """Return boolean solid mask (ny, nx), True inside the cylinder.

        Args:
            cx: Cylinder center x, normalized [0, 1].
            cy: Cylinder center y, normalized [0, 1].
            radius: Cylinder radius, normalized [0, 1].
        """
        return (self.X - cx) ** 2 + (self.Y - cy) ** 2 <= radius ** 2

    def rotate_mask(self, mask: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate obstacle mask by angle around the domain center.

        Used for the equivariance comparison experiment (Milestone 3).
        """
        from scipy.ndimage import rotate
        return rotate(mask.astype(float), angle_deg, reshape=False) > 0.5
