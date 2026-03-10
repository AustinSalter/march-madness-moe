"""Reliability diagrams and per-expert calibration analysis."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.config import CALIBRATION_N_BINS


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = CALIBRATION_N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram data).

    Args:
        y_true: Binary ground truth.
        y_pred: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        (fraction_of_positives, mean_predicted_value) arrays.
    """
    return calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")


def compute_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = CALIBRATION_N_BINS,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_b (n_b / N) * |acc_b - conf_b| where bins are uniform over [0, 1].

    Args:
        y_true: Binary ground truth.
        y_pred: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        ECE value (lower is better).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred > lo) & (y_pred <= hi)
        if lo == 0:
            mask = (y_pred >= lo) & (y_pred <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = y_pred[mask].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)

    return float(ece)


def compute_per_expert_calibration(
    y_true: np.ndarray,
    expert_preds: np.ndarray,
    names: list[str] | None = None,
    n_bins: int = CALIBRATION_N_BINS,
) -> pd.DataFrame:
    """Compute ECE for each expert.

    Args:
        y_true: Binary ground truth.
        expert_preds: (n_samples, n_experts) predictions.
        names: Expert names. Defaults to expert_0, expert_1, ...
        n_bins: Number of bins.

    Returns:
        DataFrame with columns: expert, ece.
    """
    n_experts = expert_preds.shape[1]
    if names is None:
        names = [f"expert_{i}" for i in range(n_experts)]

    rows = []
    for i, name in enumerate(names):
        ece = compute_calibration_error(y_true, expert_preds[:, i], n_bins)
        rows.append({"expert": name, "ece": ece})
    return pd.DataFrame(rows)


def compute_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expert_preds: np.ndarray | None = None,
    expert_names: list[str] | None = None,
    n_bins: int = CALIBRATION_N_BINS,
) -> dict:
    """Compute full calibration report.

    Args:
        y_true: Binary ground truth.
        y_pred: Blended predicted probabilities.
        expert_preds: Optional (n, n_experts) per-expert predictions.
        expert_names: Optional expert names.
        n_bins: Number of bins.

    Returns:
        Dict with 'blend_ece', 'calibration_curve', and optionally 'per_expert'.
    """
    report = {
        "blend_ece": compute_calibration_error(y_true, y_pred, n_bins),
    }

    frac_pos, mean_pred = compute_calibration_curve(y_true, y_pred, n_bins)
    report["calibration_curve"] = {
        "fraction_of_positives": frac_pos,
        "mean_predicted_value": mean_pred,
    }

    if expert_preds is not None:
        report["per_expert"] = compute_per_expert_calibration(
            y_true, expert_preds, expert_names, n_bins
        )

    return report
