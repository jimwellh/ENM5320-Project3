"""2D incompressible Navier-Stokes solver using NVIDIA Warp GPU kernels.

Chorin fractional-step (projection) method on a uniform Cartesian grid.
Domain [0,1]×[0,1], uniform inflow from the left (j=0).

Reynolds number convention: Re_D = U_inf * D / nu, where D = cylinder diameter.
Kinematic viscosity: nu = cylinder_diameter / re_D.
"""

import numpy as np
import warp as wp

wp.init()


# ---------------------------------------------------------------------------
# Kernel 1: Initialize velocity to uniform inflow (u=1, v=0, p=0)
# ---------------------------------------------------------------------------
@wp.kernel
def init_velocity_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    u[i, j] = 1.0
    v[i, j] = 0.0
    p[i, j] = 0.0


# ---------------------------------------------------------------------------
# Kernel 2: Advect-diffuse — central-difference advection + physical diffusion
#           + 4th-order artificial dissipation (ε₄ Δx⁴ ∂⁴/∂x⁴ + ∂⁴/∂y⁴)
# nu = cylinder_diameter / re  (cylinder-diameter-based Reynolds number)
# ---------------------------------------------------------------------------
@wp.kernel
def advect_diffuse_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    u_star: wp.array2d(dtype=wp.float32),
    v_star: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.int32),
    nx: int, ny: int,
    dx: float, dy: float,
    dt: float, re: float,
    cylinder_diameter: float,
):
    i, j = wp.tid()  # i=row(y), j=col(x)
    # Boundary cells: pass through (BCs applied separately)
    if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
        u_star[i, j] = u[i, j]
        v_star[i, j] = v[i, j]
        return
    # Solid: no-slip
    if mask[i, j] == 1:
        u_star[i, j] = 0.0
        v_star[i, j] = 0.0
        return

    # nu = D / Re_D  (kinematic viscosity scaled by cylinder diameter)
    nu = cylinder_diameter / re
    uc = u[i, j]
    vc = v[i, j]

    # Central-difference advection (2nd-order, no numerical viscosity)
    du_dx = (u[i, j + 1] - u[i, j - 1]) / (2.0 * dx)
    du_dy = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dy)
    dv_dx = (v[i, j + 1] - v[i, j - 1]) / (2.0 * dx)
    dv_dy = (v[i + 1, j] - v[i - 1, j]) / (2.0 * dy)

    # Central-difference diffusion (Laplacian)
    d2u = ((u[i, j + 1] - 2.0 * uc + u[i, j - 1]) / (dx * dx)
         + (u[i + 1, j] - 2.0 * uc + u[i - 1, j]) / (dy * dy))
    d2v = ((v[i, j + 1] - 2.0 * v[i, j] + v[i, j - 1]) / (dx * dx)
         + (v[i + 1, j] - 2.0 * v[i, j] + v[i - 1, j]) / (dy * dy))

    # 4th-order artificial dissipation: -ε₄(Δ⁴ₓu + Δ⁴ᵧu) to damp spurious oscillations
    # Only applied where the full 5-point stencil fits (2 cells from each boundary)
    eps4 = 1.0 / 64.0
    if j >= 2 and j <= nx - 3 and i >= 2 and i <= ny - 3:
        d4u = (u[i, j + 2] - 4.0 * u[i, j + 1] + 6.0 * uc - 4.0 * u[i, j - 1] + u[i, j - 2]
             + u[i + 2, j] - 4.0 * u[i + 1, j] + 6.0 * uc - 4.0 * u[i - 1, j] + u[i - 2, j])
        d4v = (v[i, j + 2] - 4.0 * v[i, j + 1] + 6.0 * v[i, j] - 4.0 * v[i, j - 1] + v[i, j - 2]
             + v[i + 2, j] - 4.0 * v[i + 1, j] + 6.0 * v[i, j] - 4.0 * v[i - 1, j] + v[i - 2, j])
        u_star[i, j] = uc + dt * (-uc * du_dx - vc * du_dy + nu * d2u) - eps4 * d4u
        v_star[i, j] = vc + dt * (-uc * dv_dx - vc * dv_dy + nu * d2v) - eps4 * d4v
    else:
        u_star[i, j] = uc + dt * (-uc * du_dx - vc * du_dy + nu * d2u)
        v_star[i, j] = vc + dt * (-uc * dv_dx - vc * dv_dy + nu * d2v)


# ---------------------------------------------------------------------------
# Kernel 3: Divergence of intermediate velocity
# ---------------------------------------------------------------------------
@wp.kernel
def divergence_kernel(
    u_star: wp.array2d(dtype=wp.float32),
    v_star: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.int32),
    nx: int, ny: int,
    dx: float, dy: float,
):
    i, j = wp.tid()
    if mask[i, j] == 1 or i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
        div[i, j] = 0.0
        return
    div[i, j] = ((u_star[i, j + 1] - u_star[i, j - 1]) / (2.0 * dx)
               + (v_star[i + 1, j] - v_star[i - 1, j]) / (2.0 * dy))


# ---------------------------------------------------------------------------
# Kernel 4: One Jacobi sweep for pressure Poisson ∇²p = div/dt
# ---------------------------------------------------------------------------
@wp.kernel
def jacobi_pressure_kernel(
    p: wp.array2d(dtype=wp.float32),
    p_new: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.int32),
    nx: int, ny: int,
    dx: float, dy: float, dt: float,
):
    i, j = wp.tid()
    if mask[i, j] == 1 or i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
        p_new[i, j] = p[i, j]
        return
    dx2 = dx * dx
    dy2 = dy * dy
    p_new[i, j] = (dy2 * (p[i, j + 1] + p[i, j - 1])
                 + dx2 * (p[i + 1, j] + p[i - 1, j])
                 - dx2 * dy2 * div[i, j] / dt) / (2.0 * (dx2 + dy2))


# ---------------------------------------------------------------------------
# Kernel 5: Velocity projection — subtract pressure gradient
# ---------------------------------------------------------------------------
@wp.kernel
def pressure_correct_kernel(
    u_star: wp.array2d(dtype=wp.float32),
    v_star: wp.array2d(dtype=wp.float32),
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.int32),
    nx: int, ny: int,
    dx: float, dy: float, dt: float,
):
    i, j = wp.tid()
    if mask[i, j] == 1:
        u[i, j] = 0.0
        v[i, j] = 0.0
        return
    if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
        u[i, j] = u_star[i, j]
        v[i, j] = v_star[i, j]
        return
    dp_dx = (p[i, j + 1] - p[i, j - 1]) / (2.0 * dx)
    dp_dy = (p[i + 1, j] - p[i - 1, j]) / (2.0 * dy)
    u[i, j] = u_star[i, j] - dt * dp_dx
    v[i, j] = v_star[i, j] - dt * dp_dy


# ---------------------------------------------------------------------------
# Kernel 6: Apply all boundary conditions in one pass
# ---------------------------------------------------------------------------
@wp.kernel
def apply_bcs_kernel(
    u: wp.array2d(dtype=wp.float32),
    v: wp.array2d(dtype=wp.float32),
    p: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.int32),
    nx: int, ny: int,
):
    i, j = wp.tid()
    # Solid: no-slip (first priority)
    if mask[i, j] == 1:
        u[i, j] = 0.0
        v[i, j] = 0.0
        return
    # Inlet j=0: uniform inflow, Neumann pressure
    if j == 0:
        u[i, j] = 1.0
        v[i, j] = 0.0
        p[i, j] = p[i, 1]
        return
    # Outlet j=nx-1: Neumann velocity, Dirichlet p=0
    if j == nx - 1:
        u[i, j] = u[i, j - 1]
        v[i, j] = v[i, j - 1]
        p[i, j] = 0.0
        return
    # Bottom wall i=0: free-slip, Neumann pressure
    if i == 0:
        u[i, j] = u[1, j]
        v[i, j] = 0.0
        p[i, j] = p[1, j]
        return
    # Top wall i=ny-1: free-slip, Neumann pressure
    if i == ny - 1:
        u[i, j] = u[ny - 2, j]
        v[i, j] = 0.0
        p[i, j] = p[ny - 2, j]
        return


# ---------------------------------------------------------------------------
# NavierStokesSolver
# ---------------------------------------------------------------------------
class NavierStokesSolver:
    """2D incompressible NS solver: Chorin projection method on uniform grid.

    Args:
        nx, ny: Grid resolution (default 128×128).
        re: Default Reynolds number Re_D (cylinder-diameter based). Can be
            overridden per call via solve(mask, re=...).
        cylinder_diameter: D = 2 * cylinder_radius. nu = D / Re_D.
        dt: Time step. 0.0002 is stable for Re_D=20–200 on 128×128.
        max_steps: Total time steps. Returns instantaneous snapshot at final step.
    """

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        re: float = 100.0,
        cylinder_diameter: float = 0.10,
        dt: float = 0.0002,
        max_steps: int = 10000,
    ):
        self.nx = nx
        self.ny = ny
        self.re = re
        self.cylinder_diameter = cylinder_diameter
        self.dt = dt
        self.max_steps = max_steps
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        self._jacobi_iters = 500  # 500 iterations for strict divergence-free constraint

        # Allocate GPU arrays once; reused across solve() calls via _reset()
        shape = (ny, nx)
        zeros = np.zeros(shape, dtype=np.float32)
        self._u      = wp.from_numpy(zeros.copy(), dtype=wp.float32, device="cuda")
        self._v      = wp.from_numpy(zeros.copy(), dtype=wp.float32, device="cuda")
        self._p      = wp.from_numpy(zeros.copy(), dtype=wp.float32, device="cuda")
        self._u_star = wp.from_numpy(zeros.copy(), dtype=wp.float32, device="cuda")
        self._v_star = wp.from_numpy(zeros.copy(), dtype=wp.float32, device="cuda")
        self._div    = wp.from_numpy(zeros.copy(), dtype=wp.float32, device="cuda")
        self._p_new  = wp.from_numpy(zeros.copy(), dtype=wp.float32, device="cuda")
        self._mask   = wp.from_numpy(
            np.zeros(shape, dtype=np.int32), dtype=wp.int32, device="cuda"
        )

    def _reset(self, geom_mask: np.ndarray) -> None:
        """Upload mask and re-initialize velocity/pressure for a new solve."""
        self._mask = wp.from_numpy(
            geom_mask.astype(np.int32), dtype=wp.int32, device="cuda"
        )
        wp.launch(init_velocity_kernel, dim=(self.ny, self.nx),
                  inputs=[self._u, self._v, self._p], device="cuda")
        wp.launch(apply_bcs_kernel, dim=(self.ny, self.nx),
                  inputs=[self._u, self._v, self._p, self._mask, self.nx, self.ny],
                  device="cuda")

    def solve(self, geom_mask: np.ndarray, re: float = None) -> dict[str, np.ndarray]:
        """Run NS solver and return instantaneous snapshot at final time step.

        Args:
            geom_mask: Boolean (ny, nx), True where solid obstacle exists.
            re: Reynolds number Re_D. Defaults to self.re set at construction.

        Returns:
            Dict with keys 'u', 'v', 'p', each shape (ny, nx).
        """
        if re is None:
            re = self.re
        self._reset(geom_mask)

        nx, ny = self.nx, self.ny
        dx, dy, dt = self.dx, self.dy, self.dt
        D = self.cylinder_diameter

        for _ in range(self.max_steps):
            # Stage 1: Advect + diffuse → u*, v*
            wp.launch(advect_diffuse_kernel, dim=(ny, nx),
                      inputs=[self._u, self._v, self._u_star, self._v_star,
                               self._mask, nx, ny, dx, dy, dt, re, D],
                      device="cuda")
            # Apply BCs to intermediate fields (u*, v*; p unchanged)
            wp.launch(apply_bcs_kernel, dim=(ny, nx),
                      inputs=[self._u_star, self._v_star, self._p,
                               self._mask, nx, ny],
                      device="cuda")

            # Stage 2: Divergence of u*, v*
            wp.launch(divergence_kernel, dim=(ny, nx),
                      inputs=[self._u_star, self._v_star, self._div,
                               self._mask, nx, ny, dx, dy],
                      device="cuda")

            # Stage 3: Pressure Poisson via Jacobi (500 iters, double-buffered)
            # 500 = even number → after loop, p_a always points back to self._p
            p_a, p_b = self._p, self._p_new
            for _ in range(self._jacobi_iters):
                wp.launch(jacobi_pressure_kernel, dim=(ny, nx),
                          inputs=[p_a, p_b, self._div, self._mask,
                                   nx, ny, dx, dy, dt],
                          device="cuda")
                wp.launch(apply_bcs_kernel, dim=(ny, nx),
                          inputs=[self._u, self._v, p_b, self._mask, nx, ny],
                          device="cuda")
                p_a, p_b = p_b, p_a  # swap

            # Stage 4: Velocity projection
            wp.launch(pressure_correct_kernel, dim=(ny, nx),
                      inputs=[self._u_star, self._v_star,
                               self._u, self._v, self._p,
                               self._mask, nx, ny, dx, dy, dt],
                      device="cuda")

            # Stage 5: BCs on corrected velocity + pressure
            wp.launch(apply_bcs_kernel, dim=(ny, nx),
                      inputs=[self._u, self._v, self._p, self._mask, nx, ny],
                      device="cuda")

        # Return instantaneous snapshot at final time step
        return {
            "u": self._u.numpy(),
            "v": self._v.numpy(),
            "p": self._p.numpy(),
        }
