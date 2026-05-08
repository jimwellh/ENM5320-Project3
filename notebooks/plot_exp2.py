"""Plot generalization experiment 2 results: Ellipse shape at Re=100."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path

from src.models.model_factory import build_model

OUT_DIR = Path("notebooks/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_fno():
    model = build_model("fno_baseline", in_channels=3, out_channels=3,
                        modes=16, width=64, n_layers=4)
    model.load_state_dict(torch.load("models/fno_baseline/best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()

def load_efno():
    model = build_model("equivariant_fno", in_channels=3, out_channels=3,
                        modes=16, width=16, n_layers=4, group="p4")
    model.load_state_dict(torch.load("models/equivariant_fno/best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()

def denormalize(tensor, stats):
    means = torch.tensor([float(stats["u_mean"]), float(stats["v_mean"]),
                          float(stats["p_mean"])]).view(1, 3, 1, 1)
    stds  = torch.tensor([float(stats["u_std"]),  float(stats["v_std"]),
                          float(stats["p_std"])]).view(1, 3, 1, 1)
    return tensor * stds + means

def load_ellipse_data():
    raw_path = "data/raw/re100.0_ellipse.npz"
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"{raw_path} not found. Did you run generate_ellipse.py?")
    
    d = np.load(raw_path)
    geom_mask = d["geom_mask"].astype(np.float32)
    re = float(d["re"])
    u = d["u"].astype(np.float32)
    v = d["v"].astype(np.float32)
    p = d["p"].astype(np.float32)

    stats = np.load("data/processed/stats.npz")
    re_range = float(stats["re_max"]) - float(stats["re_min"])
    
    H, W = geom_mask.shape
    inputs = np.zeros((3, H, W), dtype=np.float32)
    inputs[0, :, :] = geom_mask
    inputs[1, :, :] = (re - float(stats["re_min"])) / re_range
    inputs[2, :, 0] = 1.0  # inlet profile

    targets = np.zeros((3, H, W), dtype=np.float32)
    targets[0, :, :] = (u - float(stats["u_mean"])) / float(stats["u_std"])
    targets[1, :, :] = (v - float(stats["v_mean"])) / float(stats["v_std"])
    targets[2, :, :] = (p - float(stats["p_mean"])) / float(stats["p_std"])

    return torch.from_numpy(inputs), torch.from_numpy(targets), re, stats

def get_prediction(model, x, y, stats):
    with torch.no_grad():
        pred_norm = model(x.unsqueeze(0).to(DEVICE)).cpu()
    
    pred_phys = denormalize(pred_norm, stats).squeeze(0).numpy()
    gt_phys   = denormalize(y.unsqueeze(0).cpu(), stats).squeeze(0).numpy()
    return pred_phys, gt_phys

CHANNEL_LABELS = ["u-velocity", "v-velocity", "pressure"]
CHANNEL_UNITS  = ["m/s", "m/s", "Pa"]
CMAPS          = ["RdBu_r", "RdBu_r", "viridis"]

def plot_comparison(pred_fno, pred_efno, gt, re, filename):
    err_fno = np.abs(pred_fno - gt)
    err_efno = np.abs(pred_efno - gt)

    fig = plt.figure(figsize=(14, 18))
    fig.suptitle(f"Flow Field Comparison (Ellipse geometry, Re$_D$ = {re:.0f})",
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
    print("Loading models and ellipse data ...")
    x, y, re, stats = load_ellipse_data()
    
    fno = load_fno()
    efno = load_efno()

    print("Generating FNO prediction ...")
    pred_fno, gt = get_prediction(fno, x, y, stats)
    
    print("Generating EFNO prediction ...")
    pred_efno, _ = get_prediction(efno, x, y, stats)
    
    print("Plotting ...")
    plot_comparison(pred_fno, pred_efno, gt, re, "exp2_ellipse_re100.png")

if __name__ == "__main__":
    main()
