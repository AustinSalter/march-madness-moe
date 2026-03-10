"""Log-loss, accuracy, Brier score, ESPN bracket score, round-by-round breakdown."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss

from src.config import ESPN_ROUND_POINTS


def compute_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log-loss with clipping to avoid log(0)."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return log_loss(y_true, y_pred)


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute classification accuracy at a given threshold."""
    predictions = (np.asarray(y_pred) >= threshold).astype(int)
    return float(np.mean(predictions == np.asarray(y_true)))


def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    return brier_score_loss(y_true, y_pred)


def compute_espn_bracket_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rounds: np.ndarray,
) -> float:
    """Compute ESPN-style bracket score.

    Correct predictions earn points based on round:
    R1=10, R2=20, R3=40, R4=80, R5=160, R6=320.

    Args:
        y_true: Binary ground truth.
        y_pred: Predicted probabilities.
        rounds: Tournament round for each game (1-6).

    Returns:
        Total ESPN bracket score.
    """
    y_true = np.asarray(y_true)
    predictions = (np.asarray(y_pred) >= 0.5).astype(int)
    rounds = np.asarray(rounds)
    correct = predictions == y_true

    total = 0.0
    for r, pts in ESPN_ROUND_POINTS.items():
        mask = rounds == r
        total += int(correct[mask].sum()) * pts
    return total


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rounds: np.ndarray | None = None,
) -> dict:
    """Compute all metrics in one call.

    Args:
        y_true: Binary ground truth.
        y_pred: Predicted probabilities.
        rounds: Tournament round for each game. If provided, includes ESPN score.

    Returns:
        Dict of metric_name → value.
    """
    metrics = {
        "logloss": compute_logloss(y_true, y_pred),
        "accuracy": compute_accuracy(y_true, y_pred),
        "brier_score": compute_brier_score(y_true, y_pred),
    }
    if rounds is not None:
        metrics["espn_bracket_score"] = compute_espn_bracket_score(y_true, y_pred, rounds)
    return metrics


def compute_round_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rounds: np.ndarray,
) -> pd.DataFrame:
    """Compute per-round metrics table.

    Args:
        y_true: Binary ground truth.
        y_pred: Predicted probabilities.
        rounds: Tournament round for each game (1-6).

    Returns:
        DataFrame with columns: round, n_games, accuracy, logloss, brier_score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rounds = np.asarray(rounds)

    rows = []
    for r in sorted(set(rounds)):
        mask = rounds == r
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "round": int(r),
            "n_games": int(mask.sum()),
            "accuracy": compute_accuracy(yt, yp),
            "logloss": compute_logloss(yt, yp),
            "brier_score": compute_brier_score(yt, yp),
        })
    return pd.DataFrame(rows)


def compute_expert_agreement(expert_preds: np.ndarray) -> pd.DataFrame:
    """Compute pairwise Spearman correlation between expert predictions.

    Args:
        expert_preds: (n_samples, n_experts) array of predicted probabilities.

    Returns:
        DataFrame with pairwise Spearman correlation matrix.
    """
    n_experts = expert_preds.shape[1]
    names = [f"expert_{i}" for i in range(n_experts)]
    corr = np.ones((n_experts, n_experts))

    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            rho, _ = spearmanr(expert_preds[:, i], expert_preds[:, j])
            corr[i, j] = rho
            corr[j, i] = rho

    return pd.DataFrame(corr, index=names, columns=names)
