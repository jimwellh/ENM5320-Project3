"""Generic training loop compatible with PhysicsNeMo Trainer API."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader


class Trainer:
    """Trains a neural operator model.

    Args:
        model: The neural operator to train.
        optimizer: PyTorch optimizer.
        loss_fn: Loss function (DataLoss or PINOLoss).
        device: Training device string, e.g. 'cuda'.
        checkpoint_dir: Directory to save best checkpoint.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module, device: str = "cuda",
                 checkpoint_dir: str = "models/fno_baseline"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100) -> dict[str, list[float]]:
        """Run full training loop.

        Returns:
            History dict with 'train_loss' and 'val_loss' lists.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pt")

            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}", flush=True)

        return history

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y, x)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(loader)


if __name__ == "__main__":
    import json
    import argparse
    import torch
    from pathlib import Path
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.models.model_factory import build_model
    from src.training.dataset import FlowDataset
    from src.training.losses import DataLoss

    parser = argparse.ArgumentParser(description="Train FNO baseline.")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (e.g. configs/fno_baseline.yaml)")
    cli = parser.parse_args()
    cfg = OmegaConf.load(cli.config)

    model = build_model(
        cfg.model_type,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        modes=cfg.fno_modes,
        width=cfg.fno_width,
        n_layers=cfg.n_layers,
    )

    train_ds = FlowDataset("data/processed/dataset.npz", "data/splits/train_idx.npy")
    val_ds   = FlowDataset("data/processed/dataset.npz", "data/splits/val_idx.npy")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn   = DataLoss()

    trainer = Trainer(model, optimizer, loss_fn, device=cfg.device,
                      checkpoint_dir=cfg.checkpoint_dir)
    history = trainer.train(train_loader, val_loader, epochs=cfg.epochs)

    out_dir = Path(cfg.checkpoint_dir)
    with open(out_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"Training complete. Best val loss: {min(history['val_loss']):.4f}")
    print(f"Checkpoint: {out_dir / 'best.pt'}")
