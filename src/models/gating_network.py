"""PyTorch MLP gating network for MOE expert weight prediction."""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    GATING_BATCH_SIZE,
    GATING_DROPOUT,
    GATING_EPOCHS,
    GATING_HIDDEN_SIZES,
    GATING_LR,
)

logger = logging.getLogger(__name__)


class GatingMLP(nn.Module):
    """Raw MLP: context features → expert weights via softmax."""

    def __init__(
        self,
        input_dim: int,
        n_experts: int = 3,
        hidden_sizes: list[int] | None = None,
        dropout: float = GATING_DROPOUT,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or GATING_HIDDEN_SIZES
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax expert weights (n, n_experts)."""
        return torch.softmax(self.net(x), dim=1)


class GatingNetwork:
    """Training wrapper for GatingMLP.

    Trains the gating network to predict expert weights that minimize
    BCE(blended_prob, y) where blended_prob = sum(weights * expert_preds).
    Expert predictions are fixed inputs — no gradient flows through them.
    """

    def __init__(
        self,
        input_dim: int = 4,
        n_experts: int = 3,
        lr: float = GATING_LR,
        epochs: int = GATING_EPOCHS,
        batch_size: int = GATING_BATCH_SIZE,
    ):
        self.input_dim = input_dim
        self.n_experts = n_experts
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.mlp = GatingMLP(input_dim, n_experts)
        self.device = torch.device("cpu")
        self._fitted = False

    def fit(
        self,
        context_X: np.ndarray,
        expert_preds: np.ndarray,
        y: np.ndarray,
        patience: int = 20,
        val_fraction: float = 0.2,
    ) -> "GatingNetwork":
        """Train gating network.

        Args:
            context_X: (n, input_dim) gating features.
            expert_preds: (n, n_experts) fixed expert predictions.
            y: (n,) binary labels.
            patience: Early stopping patience.
            val_fraction: Fraction for holdout validation.

        Returns:
            self
        """
        n = len(y)
        idx = np.random.RandomState(42).permutation(n)
        val_size = int(n * val_fraction)
        val_idx, train_idx = idx[:val_size], idx[val_size:]

        def to_tensor(arr):
            return torch.tensor(np.asarray(arr), dtype=torch.float32)

        ctx_train = to_tensor(context_X[train_idx])
        ep_train = to_tensor(expert_preds[train_idx])
        y_train = to_tensor(y[train_idx]).unsqueeze(1)
        ctx_val = to_tensor(context_X[val_idx])
        ep_val = to_tensor(expert_preds[val_idx])
        y_val = to_tensor(y[val_idx]).unsqueeze(1)

        train_ds = TensorDataset(ctx_train, ep_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.mlp.to(self.device)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.epochs):
            # Train
            self.mlp.train()
            for ctx_b, ep_b, y_b in train_loader:
                weights = self.mlp(ctx_b)  # (batch, n_experts)
                blended = (weights * ep_b).sum(dim=1, keepdim=True)  # (batch, 1)
                blended = blended.clamp(1e-7, 1 - 1e-7)
                loss = loss_fn(blended, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate
            self.mlp.eval()
            with torch.no_grad():
                val_weights = self.mlp(ctx_val)
                val_blended = (val_weights * ep_val).sum(dim=1, keepdim=True)
                val_blended = val_blended.clamp(1e-7, 1 - 1e-7)
                val_loss = loss_fn(val_blended, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.mlp.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

        if best_state is not None:
            self.mlp.load_state_dict(best_state)

        self._fitted = True
        logger.info(
            "Trained gating network: best_val_loss=%.4f, epochs=%d",
            best_val_loss, epoch + 1,
        )
        return self

    def predict_weights(self, context_X: np.ndarray) -> np.ndarray:
        """Return (n, n_experts) gating weights.

        Args:
            context_X: (n, input_dim) gating features.

        Returns:
            (n, n_experts) array of weights summing to 1 per row.
        """
        if not self._fitted:
            raise RuntimeError("Must fit() before predict_weights()")

        self.mlp.eval()
        with torch.no_grad():
            ctx = torch.tensor(np.asarray(context_X), dtype=torch.float32).to(self.device)
            weights = self.mlp(ctx).cpu().numpy()
        return weights

    def save(self, path: str | Path) -> None:
        """Save gating network to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "input_dim": self.input_dim,
            "n_experts": self.n_experts,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "mlp_state_dict": self.mlp.state_dict(),
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved gating network to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GatingNetwork":
        """Load a saved gating network."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        gn = cls(
            input_dim=state["input_dim"],
            n_experts=state["n_experts"],
            lr=state["lr"],
            epochs=state["epochs"],
            batch_size=state["batch_size"],
        )
        gn.mlp.load_state_dict(state["mlp_state_dict"])
        gn._fitted = state["fitted"]
        logger.info("Loaded gating network from %s", path)
        return gn
