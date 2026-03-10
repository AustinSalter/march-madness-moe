"""XGBoost binary classifier wrapper with sample weighting support."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import (
    EARLY_STOPPING_ROUNDS,
    EFFICIENCY_WEIGHT_FLOOR,
    EXPERT_TYPES,
    OPTUNA_N_TRIALS,
    OPTUNA_SEARCH_SPACE,
    OPTUNA_TIMEOUT,
    UNCERTAINTY_SAME_SEED_WEIGHT,
    XGBOOST_PARAMS,
)
from src.features.pipeline import FeatureSet

logger = logging.getLogger(__name__)


class TreeExpert:
    """XGBoost expert with per-type sample weighting and post-hoc calibration."""

    def __init__(
        self,
        expert_type: str = "seed_baseline",
        params: dict | None = None,
    ):
        if expert_type not in EXPERT_TYPES:
            raise ValueError(f"Unknown expert_type {expert_type!r}. Must be one of {EXPERT_TYPES}")
        self.expert_type = expert_type
        self.params = {**XGBOOST_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None
        self.calibrator: CalibratedClassifierCV | None = None
        self._is_calibrated = False

    def compute_sample_weights(self, fs: FeatureSet) -> np.ndarray:
        """Compute per-sample weights based on expert type.

        Args:
            fs: FeatureSet containing X, y, meta, ranking_targets.

        Returns:
            1-D array of weights, normalized so sum == N.
        """
        n = len(fs.y)

        if self.expert_type == "seed_baseline":
            return np.ones(n)

        elif self.expert_type == "efficiency_delta":
            if "adjem_delta" not in fs.X.columns:
                logger.warning("adjem_delta not in X — using uniform weights")
                return np.ones(n)
            w = fs.X["adjem_delta"].abs().values.copy()
            w = np.maximum(w, EFFICIENCY_WEIGHT_FLOOR)
            w = w * (n / w.sum())
            return w

        elif self.expert_type == "uncertainty_calibration":
            if "seed_diff" not in fs.X.columns:
                logger.warning("seed_diff not in X — using uniform weights")
                return np.ones(n)
            seed_diff = fs.X["seed_diff"].abs().values.astype(float)
            eps = 1.0
            w = 1.0 / (seed_diff + eps)
            # Same-seed matchups (seed_diff == 0) get boosted weight
            same_seed = seed_diff == 0
            w[same_seed] = UNCERTAINTY_SAME_SEED_WEIGHT
            w = w * (n / w.sum())
            return w

        return np.ones(n)

    def fit(
        self,
        fs: FeatureSet,
        val_fs: FeatureSet | None = None,
    ) -> "TreeExpert":
        """Train XGBoost with sample weights and early stopping.

        Args:
            fs: Training FeatureSet.
            val_fs: Validation FeatureSet. If None, uses 80/20 stratified split.

        Returns:
            self
        """
        weights = self.compute_sample_weights(fs)

        # Prepare params for XGBClassifier
        fit_params = {**self.params}
        n_estimators = fit_params.pop("n_estimators", 1000)
        random_state = fit_params.pop("random_state", 42)

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **fit_params,
        )

        if val_fs is not None:
            val_weights = self.compute_sample_weights(val_fs)
            self.model.fit(
                fs.X, fs.y,
                sample_weight=weights,
                eval_set=[(val_fs.X, val_fs.y)],
                sample_weight_eval_set=[val_weights],
                verbose=False,
            )
        else:
            # 80/20 stratified split for early stopping
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(fs.X, fs.y))
            X_train, X_val = fs.X.iloc[train_idx], fs.X.iloc[val_idx]
            y_train, y_val = fs.y.iloc[train_idx], fs.y.iloc[val_idx]
            w_train, w_val = weights[train_idx], weights[val_idx]

            self.model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[w_val],
                verbose=False,
            )

        best_iter = getattr(self.model, "best_iteration", None)
        logger.info(
            "Trained %s expert: best_iteration=%s, n_features=%d, n_samples=%d",
            self.expert_type, best_iter, fs.X.shape[1], fs.X.shape[0],
        )
        return self

    def calibrate(
        self,
        fs: FeatureSet,
        method: str = "isotonic",
        cv: int = 5,
    ) -> "TreeExpert":
        """Post-hoc calibration via CalibratedClassifierCV.

        Args:
            fs: FeatureSet to use for calibration fitting.
            method: 'isotonic' or 'sigmoid'.
            cv: Number of cross-validation folds.

        Returns:
            self
        """
        if self.model is None:
            raise RuntimeError("Must fit() before calibrate()")

        self.calibrator = CalibratedClassifierCV(
            self.model,
            method=method,
            cv=cv,
        )
        weights = self.compute_sample_weights(fs)
        self.calibrator.fit(fs.X, fs.y, sample_weight=weights)
        self._is_calibrated = True

        logger.info("Calibrated %s expert with method=%s, cv=%d", self.expert_type, method, cv)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return 1-D P(higher_seed_wins).

        Uses calibrator if fitted, otherwise raw model.

        Args:
            X: Feature matrix.

        Returns:
            1-D array of probabilities.
        """
        if self.model is None:
            raise RuntimeError("Must fit() before predict_proba()")

        if self._is_calibrated and self.calibrator is not None:
            proba = self.calibrator.predict_proba(X)
        else:
            proba = self.model.predict_proba(X)

        # Return probability of class 1 (higher_seed_wins)
        return proba[:, 1]

    def get_feature_importance(self) -> pd.Series:
        """Gain-based feature importances.

        Returns:
            Series with feature names as index, gain values as values.
        """
        if self.model is None:
            raise RuntimeError("Must fit() before get_feature_importance()")

        imp = self.model.get_booster().get_score(importance_type="gain")
        return pd.Series(imp).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        """Pickle-serialize the expert to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved %s expert to %s", self.expert_type, path)

    @classmethod
    def load(cls, path: str | Path) -> "TreeExpert":
        """Load a pickled TreeExpert."""
        with open(path, "rb") as f:
            expert = pickle.load(f)
        if not isinstance(expert, cls):
            raise TypeError(f"Expected TreeExpert, got {type(expert)}")
        logger.info("Loaded %s expert from %s", expert.expert_type, path)
        return expert


def tune_expert_hyperparams(
    fs: FeatureSet,
    expert_type: str = "seed_baseline",
    val_fs: FeatureSet | None = None,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
) -> dict:
    """Tune XGBoost hyperparameters using Optuna.

    Args:
        fs: Training FeatureSet.
        expert_type: Expert type for sample weighting.
        val_fs: Validation FeatureSet. If None, uses stratified split.
        n_trials: Number of Optuna trials.
        timeout: Timeout in seconds.

    Returns:
        Best params dict (only the tuned params, not full XGBOOST_PARAMS).
    """
    import optuna
    from sklearn.metrics import log_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Prepare train/val split if no val_fs
    dummy_expert = TreeExpert(expert_type=expert_type)
    weights = dummy_expert.compute_sample_weights(fs)

    if val_fs is not None:
        X_train, y_train, w_train = fs.X, fs.y, weights
        X_val, y_val = val_fs.X, val_fs.y
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(fs.X, fs.y))
        X_train, X_val = fs.X.iloc[train_idx], fs.X.iloc[val_idx]
        y_train, y_val = fs.y.iloc[train_idx], fs.y.iloc[val_idx]
        w_train = weights[train_idx]

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for param_name, (lo, hi) in OPTUNA_SEARCH_SPACE.items():
            if isinstance(lo, int) and isinstance(hi, int):
                params[param_name] = trial.suggest_int(param_name, lo, hi)
            else:
                params[param_name] = trial.suggest_float(param_name, lo, hi)

        clf = xgb.XGBClassifier(
            n_estimators=1000,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            **params,
        )
        clf.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = clf.predict_proba(X_val)[:, 1]
        return log_loss(y_val, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    logger.info(
        "Optuna %s: best logloss=%.4f after %d trials",
        expert_type, study.best_value, len(study.trials),
    )
    return study.best_params
