"""Milestone 2 visualization: loss curve + flow field comparisons at Re=20 and Re=200."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path

from src.models.model_factory import build_model
from src.training.dataset import FlowDataset
from src.training.metrics import relative_l2_error

OUT_DIR = Path("notebooks/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_model():
    model = build_model("fno_baseline", in_channels=3, out_channels=3,
                        modes=16, width=64, n_layers=4)
    model.load_state_dict(torch.load("models/fno_baseline/best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()


def denormalize(tensor, stats):
    """Convert Z-score normalized (N,3,H,W) tensor back to physical units."""
    means = torch.tensor([float(stats["u_mean"]), float(stats["v_mean"]),
                          float(stats["p_mean"])]).view(1, 3, 1, 1)
    stds  = torch.tensor([float(stats["u_std"]),  float(stats["v_std"]),
                          float(stats["p_std"])]).view(1, 3, 1, 1)
    return tensor * stds + means


def get_prediction(model, dataset, idx, stats):
    """Return (pred_phys, gt_phys, Re) all in physical units, as numpy arrays."""
    x, y = dataset[idx]
    re = float(x[1, 0, 0]) * 180.0 + 20.0  # denormalize Re from channel 1

    with torch.no_grad():
        pred_norm = model(x.unsqueeze(0).to(DEVICE)).cpu()

    pred_phys = denormalize(pred_norm,            stats).squeeze(0).numpy()   # (3,H,W)
    gt_phys   = denormalize(y.unsqueeze(0).cpu(), stats).squeeze(0).numpy()   # (3,H,W)
    return pred_phys, gt_phys, re


# ── Plot 1: training loss curve ───────────────────────────────────────────────

def plot_loss_curve():
    with open("models/fno_baseline/history.json") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    epochs     = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="Train loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_loss,   label="Val loss",   color="#FF5722", linewidth=1.5)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="5% target")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 loss (normalized space)")
    ax.set_title("FNO Baseline — Training & Validation Loss")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, len(train_loss))

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val   = min(val_loss)
    ax.axvline(best_epoch, color="#FF5722", linestyle=":", linewidth=1, alpha=0.6)
    ax.annotate(f"Best val={best_val:.4f}\n@ epoch {best_epoch}",
                xy=(best_epoch, best_val), xytext=(best_epoch + 5, best_val * 1.5),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "1_loss_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2 & 3: flow field comparison ────────────────────────────────────────

CHANNEL_LABELS = ["u-velocity", "v-velocity", "pressure"]
CHANNEL_UNITS  = ["m/s", "m/s", "Pa"]
CMAPS          = ["RdBu_r", "RdBu_r", "viridis"]


def plot_comparison(pred, gt, re, filename):
    """3×3 grid: rows = [Ground Truth, Predicted, |Error|], cols = [u, v, p]."""
    error = np.abs(pred - gt)

    # per-channel relative L2 for subtitle
    per_ch_err = [
        np.linalg.norm(pred[c] - gt[c]) / (np.linalg.norm(gt[c]) + 1e-8)
        for c in range(3)
    ]

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(f"FNO Baseline — Flow Field Comparison  (Re$_D$ = {re:.0f})",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    row_labels = ["Ground Truth", "FNO Predicted", "Absolute Error"]

    for row, (data_row, row_label) in enumerate(zip([gt, pred, error], row_labels)):
        for col, (ch_label, unit, cmap) in enumerate(
                zip(CHANNEL_LABELS, CHANNEL_UNITS, CMAPS)):
            ax = fig.add_subplot(gs[row, col])
            field = data_row[col]

            if row < 2:
                # GT and Pred: share colour scale based on GT range
                vmax = np.abs(gt[col]).max()
                vmin = -vmax if cmap == "RdBu_r" else gt[col].min()
                im = ax.imshow(field, origin="lower", cmap=cmap,
                               vmin=vmin, vmax=vmax)
            else:
                # Error: 0 → max_error
                im = ax.imshow(field, origin="lower", cmap="hot_r",
                               vmin=0, vmax=error[col].max())

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=unit if row < 2 else f"Δ{unit}")
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(f"{ch_label}\n(rel. L2 = {per_ch_err[col]*100:.2f}%)",
                             fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=6)

    out = OUT_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and data …")
    stats   = np.load("data/processed/stats.npz")
    test_ds = FlowDataset("data/processed/dataset.npz", "data/splits/test_idx.npy")
    model   = load_model()

    # Find test-set samples closest to target Re values
    re_vals = (test_ds.inputs[:, 1, 0, 0] * 180 + 20).numpy()
    idx_re20  = int(np.argmin(np.abs(re_vals - 20.0)))
    idx_re200 = int(np.argmin(np.abs(re_vals - 200.0)))
    print(f"Re=20  sample: index={idx_re20},  actual Re={re_vals[idx_re20]:.2f}")
    print(f"Re=200 sample: index={idx_re200}, actual Re={re_vals[idx_re200]:.2f}")

    # 1. Loss curve
    print("\nPlotting loss curve …")
    plot_loss_curve()

    # 2. Re ≈ 20
    print("Plotting Re=20 comparison …")
    pred20, gt20, re20 = get_prediction(model, test_ds, idx_re20, stats)
    plot_comparison(pred20, gt20, re20, "2_flow_comparison_Re20.png")

    # 3. Re ≈ 200
    print("Plotting Re=200 comparison …")
    pred200, gt200, re200 = get_prediction(model, test_ds, idx_re200, stats)
    plot_comparison(pred200, gt200, re200, "3_flow_comparison_Re200.png")

    print(f"\nAll figures saved to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
