"""PyTorch multi-head network with torchsort differentiable Spearman loss."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import BACKBONE_DROPOUT, BACKBONE_HIDDEN_SIZES, EXPERT_TYPES
from src.features.pipeline import FeatureSet
from src.models.ranking_loss import MultiTaskRankingLoss

logger = logging.getLogger(__name__)


class SharedBackbone(nn.Module):
    """Shared feature extractor: Linear(n_features→32) → ReLU → Dropout → Linear(32→24) → ReLU."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = BACKBONE_DROPOUT,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or BACKBONE_HIDDEN_SIZES
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertHead(nn.Module):
    """Single expert head: Linear(backbone_dim→1) → Sigmoid."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class MultiTaskExperts(nn.Module):
    """Multi-head network: shared backbone + 3 expert heads.

    Drop-in replacement for 3 TreeExperts in MOEEnsemble.
    Each head has its own MultiTaskRankingLoss (BCE + Spearman).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = BACKBONE_DROPOUT,
        n_experts: int = 3,
    ):
        super().__init__()
        self.backbone = SharedBackbone(input_dim, hidden_sizes, dropout)
        self.heads = nn.ModuleList([
            ExpertHead(self.backbone.output_dim) for _ in range(n_experts)
        ])
        self.n_experts = n_experts
        self._fitted = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of (n,) predictions, one per expert head."""
        shared = self.backbone(x)
        return [head(shared) for head in self.heads]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return (n, n_experts) predictions.

        Args:
            X: Feature matrix.

        Returns:
            (n, n_experts) array of predictions.
        """
        if not self._fitted:
            raise RuntimeError("Must fit() before predict_proba()")

        self.eval()
        with torch.no_grad():
            x = torch.tensor(np.asarray(X, dtype=np.float32))
            preds = self.forward(x)
            return np.column_stack([p.numpy() for p in preds])


class MultiTaskExpertsTrainer:
    """Training wrapper for MultiTaskExperts."""

    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        epochs: int = 300,
        ranking_lambda: float = 0.3,
    ):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.ranking_lambda = ranking_lambda
        self.model = MultiTaskExperts(input_dim)
        self.losses: list[MultiTaskRankingLoss] = [
            MultiTaskRankingLoss(ranking_lambda=ranking_lambda, expert_idx=i)
            for i in range(3)
        ]

    def fit(
        self,
        fs: FeatureSet,
        val_fs: FeatureSet | None = None,
        patience: int = 30,
    ) -> MultiTaskExperts:
        """Train multi-task experts.

        Uses full-batch training for Spearman stability on ~1,260 samples.

        Args:
            fs: Training FeatureSet.
            val_fs: Optional validation FeatureSet.
            patience: Early stopping patience.

        Returns:
            Trained MultiTaskExperts model.
        """
        # Ranking target columns match EXPERT_TYPES order
        ranking_target_cols = [
            "seed_implied_prob",
            "efficiency_delta_rank",
            "game_certainty_score",
        ]

        X = torch.tensor(fs.X.values, dtype=torch.float32)
        y = torch.tensor(fs.y.values, dtype=torch.float32)
        ranking_targets = [
            torch.tensor(fs.ranking_targets[col].values, dtype=torch.float32)
            for col in ranking_target_cols
        ]

        if val_fs is not None:
            X_val = torch.tensor(val_fs.X.values, dtype=torch.float32)
            y_val = torch.tensor(val_fs.y.values, dtype=torch.float32)
            rt_val = [
                torch.tensor(val_fs.ranking_targets[col].values, dtype=torch.float32)
                for col in ranking_target_cols
            ]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            # Full-batch training
            self.model.train()
            preds = self.model(X)
            total_loss = torch.tensor(0.0)
            for i in range(3):
                total_loss = total_loss + self.losses[i](preds[i], y, ranking_targets[i])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Validation
            if val_fs is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val)
                    val_loss = sum(
                        self.losses[i](val_preds[i], y_val, rt_val[i])
                        for i in range(3)
                    )
                    val_loss_val = val_loss.item()

                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model._fitted = True
        logger.info("Trained MultiTaskExperts: %d epochs, best_val_loss=%.4f", epoch + 1, best_val_loss)
        return self.model

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "ranking_lambda": self.ranking_lambda,
            "fitted": self.model._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved MultiTaskExperts to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MultiTaskExpertsTrainer":
        """Load saved model."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        trainer = cls(
            input_dim=state["input_dim"],
            lr=state["lr"],
            epochs=state["epochs"],
            ranking_lambda=state["ranking_lambda"],
        )
        trainer.model.load_state_dict(state["model_state_dict"])
        trainer.model._fitted = state["fitted"]
        logger.info("Loaded MultiTaskExperts from %s", path)
        return trainer
