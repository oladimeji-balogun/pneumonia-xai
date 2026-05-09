import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import BaseModel


class Trainer:
    """
    Training loop for benchmarking CNN architectures on chest X-ray classification.

    Handles:
    - Weighted loss for class imbalance
    - Early stopping to prevent overfitting
    - Checkpoint saving (best model by val loss)
    - Per-epoch metrics logging

    Args:
        model: Any BaseModel subclass.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        class_weights: Per-class loss weights (from dataset). Shape: (num_classes,)
        learning_rate: Initial learning rate. Default: 1e-3.
        epochs: Maximum training epochs. Default: 30.
        patience: Early stopping patience (epochs without val loss improvement). Default: 5.
        checkpoint_dir: Directory to save best model checkpoint. Default: experiments/results.
        device: torch device. Auto-detected if None.
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.FloatTensor,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        patience: int = 5,
        checkpoint_dir: str | Path = "experiments/results",
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for imgs, labels in tqdm(loader, leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

                if training:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

        return {
            "loss": total_loss / total,
            "acc": correct / total,
        }

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Training history dict with keys: train_loss, val_loss, train_acc, val_acc.
        """
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        checkpoint_path = self.checkpoint_dir / f"{self.model.model_name}_best.pth"

        print(f"\nTraining {self.model.model_name} on {self.device}")
        print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc':<12} {'LR'}")
        print("-" * 75)

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)

            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_acc"].append(val_metrics["acc"])

            elapsed = time.time() - start
            print(
                f"{epoch:<8} {train_metrics['loss']:<14.4f} {train_metrics['acc']:<14.4f} "
                f"{val_metrics['loss']:<14.4f} {val_metrics['acc']:<12.4f} {current_lr:.2e}  ({elapsed:.1f}s)"
            )

            # Checkpoint best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_acc": val_metrics["acc"],
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} — no improvement for {self.patience} epochs.")
                    break

        print(f"\nBest val loss: {best_val_loss:.4f} | Checkpoint: {checkpoint_path}")
        return self.history