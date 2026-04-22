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
                print(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")

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
