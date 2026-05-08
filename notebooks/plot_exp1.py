"""Plot generalization experiment 1 results: cy lower-half train, upper-half test."""

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

def load_fno():
    model = build_model("fno_baseline", in_channels=3, out_channels=3,
                        modes=16, width=64, n_layers=4)
    model.load_state_dict(torch.load("models/exp1_fno/best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()

def load_efno():
    model = build_model("equivariant_fno", in_channels=3, out_channels=3,
                        modes=16, width=16, n_layers=4, group="p4")
    model.load_state_dict(torch.load("models/exp1_efno/best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()

def denormalize(tensor, stats):
    means = torch.tensor([float(stats["u_mean"]), float(stats["v_mean"]),
                          float(stats["p_mean"])]).view(1, 3, 1, 1)
    stds  = torch.tensor([float(stats["u_std"]),  float(stats["v_std"]),
                          float(stats["p_std"])]).view(1, 3, 1, 1)
    return tensor * stds + means

def get_prediction(model, dataset, idx, stats):
    x, y = dataset[idx]
    re = float(x[1, 0, 0]) * 180.0 + 20.0
    
    with torch.no_grad():
        pred_norm = model(x.unsqueeze(0).to(DEVICE)).cpu()
    
    pred_phys = denormalize(pred_norm, stats).squeeze(0).numpy()
    gt_phys   = denormalize(y.unsqueeze(0).cpu(), stats).squeeze(0).numpy()
    return pred_phys, gt_phys, re

def plot_loss_curves():
    with open("models/exp1_fno/history.json") as f:
        fno_history = json.load(f)
    with open("models/exp1_efno/history.json") as f:
        efno_history = json.load(f)

    epochs = range(1, len(fno_history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, fno_history["val_loss"], label="FNO Val Loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, efno_history["val_loss"], label="EFNO Val Loss", color="#FF5722", linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 loss (normalized space)")
    ax.set_title("Validation Loss: cy > 0.5 generalisation")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, len(epochs))

    plt.tight_layout()
    out = OUT_DIR / "exp1_loss_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

CHANNEL_LABELS = ["u-velocity", "v-velocity", "pressure"]
CHANNEL_UNITS  = ["m/s", "m/s", "Pa"]
CMAPS          = ["RdBu_r", "RdBu_r", "viridis"]

def plot_comparison(pred_fno, pred_efno, gt, re, filename):
    err_fno = np.abs(pred_fno - gt)
    err_efno = np.abs(pred_efno - gt)

    fig = plt.figure(figsize=(14, 18))
    fig.suptitle(f"Flow Field Comparison (cy > 0.5 test, Re$_D$ = {re:.0f})",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35)
    row_labels = ["Ground Truth", "FNO Predicted", "FNO Absolute Error", "EFNO Predicted", "EFNO Absolute Error"]
    data_rows = [gt, pred_fno, err_fno, pred_efno, err_efno]

    for row, (data_row, row_label) in enumerate(zip(data_rows, row_labels)):
        for col, (ch_label, unit, cmap) in enumerate(zip(CHANNEL_LABELS, CHANNEL_UNITS, CMAPS)):
            ax = fig.add_subplot(gs[row, col])
            field = data_row[col]

            if row in [0, 1, 3]:
                vmax = np.abs(gt[col]).max()
                vmin = -vmax if cmap == "RdBu_r" else gt[col].min()
                im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                max_err = max(err_fno[col].max(), err_efno[col].max())
                im = ax.imshow(field, origin="lower", cmap="hot_r", vmin=0, vmax=max_err)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit if row in [0, 1, 3] else f"Δ{unit}")
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(ch_label, fontsize=10)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=6)

    out = OUT_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

def main():
    print("Loading models and data ...")
    stats = np.load("data/processed/stats.npz")
    test_ds = FlowDataset("data/processed/dataset.npz", "data/splits/test_cy_upper.npy")
    
    fno = load_fno()
    efno = load_efno()

    re_vals = (test_ds.inputs[:, 1, 0, 0] * 180 + 20).numpy()
    idx_re20  = int(np.argmin(np.abs(re_vals - 20.0)))
    idx_re200 = int(np.argmin(np.abs(re_vals - 200.0)))

    print("Plotting loss curves ...")
    plot_loss_curves()

    print("Plotting Re=20 comparison ...")
    pred_fno20, gt20, re20 = get_prediction(fno, test_ds, idx_re20, stats)
    pred_efno20, _, _ = get_prediction(efno, test_ds, idx_re20, stats)
    plot_comparison(pred_fno20, pred_efno20, gt20, re20, "exp1_flow_compare_re20.png")

    print("Plotting Re=200 comparison ...")
    pred_fno200, gt200, re200 = get_prediction(fno, test_ds, idx_re200, stats)
    pred_efno200, _, _ = get_prediction(efno, test_ds, idx_re200, stats)
    plot_comparison(pred_fno200, pred_efno200, gt200, re200, "exp1_flow_compare_re200.png")

if __name__ == "__main__":
    main()
