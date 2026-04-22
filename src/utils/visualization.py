"""Flow field visualization utilities for training monitoring."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_flow_field(u: np.ndarray, v: np.ndarray, p: np.ndarray,
                   title: str = "Flow Field", save_path: str | None = None) -> None:
    """Plot u, v, p fields side by side with streamlines.

    Args:
        u: Velocity x-component (H, W).
        v: Velocity y-component (H, W).
        p: Pressure field (H, W).
        title: Figure title.
        save_path: If provided, save figure instead of showing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)

    for ax, field, label in zip(axes, [u, v, p], ["u (x-velocity)", "v (y-velocity)", "p (pressure)"]):
        im = ax.imshow(field, origin="lower", cmap="RdBu_r")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

    # Overlay streamlines on velocity magnitude
    h, w = u.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    axes[0].streamplot(x, y, u, v, color="k", density=1.0, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
