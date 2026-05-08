"""Geometry utilities: cylinder mask generation, obstacle parameterization, and field rotation."""

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

    def make_ellipse_mask(self, cx: float, cy: float, rx: float = 0.05, ry: float = 0.025) -> np.ndarray:
        """Return boolean solid mask (ny, nx), True inside the ellipse.

        Args:
            cx: Ellipse center x, normalized [0, 1].
            cy: Ellipse center y, normalized [0, 1].
            rx: Ellipse semi-major axis, normalized [0, 1].
            ry: Ellipse semi-minor axis, normalized [0, 1].
        """
        return ((self.X - cx) / rx) ** 2 + ((self.Y - cy) / ry) ** 2 <= 1.0

    def rotate_mask(self, mask: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate obstacle mask by angle around the domain center.

        Used for the equivariance comparison experiment (Milestone 3).
        """
        from scipy.ndimage import rotate
        return rotate(mask.astype(float), angle_deg, reshape=False) > 0.5


def rotate_flow_field(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    angle_deg: float,
) -> tuple:
    """Rotate a flow field by angle_deg (CCW), matching rotate_mask convention.

    Applies both spatial grid rotation and 2D rotation-matrix transform to
    velocity components so the result is physically consistent.  Pressure is a
    scalar field and only requires the spatial rotation.

    Positive angle_deg is CCW, identical to scipy.ndimage.rotate convention.

    Args:
        u, v: Velocity components, shape (ny, nx).
        p:    Pressure field, shape (ny, nx).
        angle_deg: CCW rotation angle in degrees.

    Returns:
        (u_rot, v_rot, p_rot): Rotated fields, same shape as inputs.
    """
    from scipy.ndimage import rotate as _rotate

    # Step 1: spatial rotation of every field component
    kw = dict(reshape=False)
    u_s = _rotate(u.astype(float), angle_deg, **kw)
    v_s = _rotate(v.astype(float), angle_deg, **kw)
    p_rot = _rotate(p.astype(float), angle_deg, **kw)

    # Step 2: apply 2-D rotation matrix to velocity vector components
    # CCW by theta: u' = cos*u - sin*v,  v' = sin*u + cos*v
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u_rot = cos_t * u_s - sin_t * v_s
    v_rot = sin_t * u_s + cos_t * v_s

    return u_rot, v_rot, p_rot
