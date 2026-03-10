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
    EXPERT_FEATURE_SUBSETS,
    EXPERT_RANKING_TARGET,
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
        self.feature_columns: list[str] | None = None
        self.use_feature_subset: bool = True

    def _prepare_X(self, X: pd.DataFrame, ranking_targets: pd.DataFrame | None = None) -> pd.DataFrame:
        """Inject ranking target and apply feature subsetting.

        First call (feature_columns is None) resolves the config subset.
        Subsequent calls reuse the stored feature_columns list.
        """
        # Step 1: Inject ranking target column if available
        target_col = EXPERT_RANKING_TARGET.get(self.expert_type)
        if target_col and ranking_targets is not None and target_col in ranking_targets.columns:
            if target_col not in X.columns:
                X = X.copy()
                X[target_col] = ranking_targets[target_col].values

        # Step 2: Apply feature subsetting
        if self.feature_columns is not None:
            return X[self.feature_columns]

        if not self.use_feature_subset:
            self.feature_columns = list(X.columns)
            return X

        configured = EXPERT_FEATURE_SUBSETS.get(self.expert_type)
        if configured is None:
            self.feature_columns = list(X.columns)
            return X

        available = [c for c in configured if c in X.columns]
        if not available:
            logger.warning("%s: no configured features found — using all", self.expert_type)
            self.feature_columns = list(X.columns)
            return X

        self.feature_columns = available
        return X[available]

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
            weight_col = "net_rating_delta"
            if weight_col not in fs.X.columns:
                weight_col = "adjem_delta"  # fallback
            if weight_col not in fs.X.columns:
                logger.warning("No efficiency weighting column — using uniform")
                return np.ones(n)
            w = fs.X[weight_col].abs().values.copy()
            w = np.nan_to_num(w, nan=0.0)
            w = np.maximum(w, EFFICIENCY_WEIGHT_FLOOR)
            w = w * (n / w.sum())
            return w

        elif self.expert_type == "uncertainty_calibration":
            # 1/|seed_diff| — focus on close games where uncertainty matters most
            if "seed_diff" not in fs.X.columns:
                return np.ones(n)
            seed_diff = fs.X["seed_diff"].abs().values.astype(float)
            w = 1.0 / (seed_diff + 1.0)
            w[seed_diff == 0] = UNCERTAINTY_SAME_SEED_WEIGHT
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
        X_train = self._prepare_X(fs.X, fs.ranking_targets)  # Sets self.feature_columns on first call

        # Auto-adjust colsample_bytree for small feature sets
        if len(self.feature_columns) <= 6:
            self.params["colsample_bytree"] = 1.0

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
            X_val = self._prepare_X(val_fs.X, val_fs.ranking_targets)
            self.model.fit(
                X_train, fs.y,
                sample_weight=weights,
                eval_set=[(X_val, val_fs.y)],
                sample_weight_eval_set=[val_weights],
                verbose=False,
            )
        else:
            # 80/20 stratified split for early stopping
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(X_train, fs.y))
            self.model.fit(
                X_train.iloc[train_idx], fs.y.iloc[train_idx],
                sample_weight=weights[train_idx],
                eval_set=[(X_train.iloc[val_idx], fs.y.iloc[val_idx])],
                sample_weight_eval_set=[weights[val_idx]],
                verbose=False,
            )

        best_iter = getattr(self.model, "best_iteration", None)
        logger.info(
            "Trained %s expert: best_iteration=%s, n_features=%d, n_samples=%d",
            self.expert_type, best_iter, len(self.feature_columns), fs.X.shape[0],
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
        X_cal = self._prepare_X(fs.X, fs.ranking_targets)
        self.calibrator.fit(X_cal, fs.y, sample_weight=weights)
        self._is_calibrated = True

        logger.info("Calibrated %s expert with method=%s, cv=%d", self.expert_type, method, cv)
        return self

    def predict_proba(self, X: "pd.DataFrame | np.ndarray | FeatureSet") -> np.ndarray:
        """Return 1-D P(higher_seed_wins).

        Uses calibrator if fitted, otherwise raw model.

        Args:
            X: Feature matrix or FeatureSet.

        Returns:
            1-D array of probabilities.
        """
        if self.model is None:
            raise RuntimeError("Must fit() before predict_proba()")

        if isinstance(X, FeatureSet):
            X_input = self._prepare_X(X.X, X.ranking_targets)
        elif self.feature_columns is not None and isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Raw DataFrame is missing columns {missing} that were injected from "
                    f"ranking_targets during fit(). Pass the full FeatureSet instead of .X"
                )
            X_input = X[self.feature_columns]
        else:
            X_input = X

        if self._is_calibrated and self.calibrator is not None:
            proba = self.calibrator.predict_proba(X_input)
        else:
            proba = self.model.predict_proba(X_input)

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
        # Pickle compat for models saved before feature subsetting
        if not hasattr(expert, "feature_columns"):
            expert.feature_columns = None
        if not hasattr(expert, "use_feature_subset"):
            expert.use_feature_subset = True
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
    X_full = dummy_expert._prepare_X(fs.X, fs.ranking_targets)
    feature_cols = dummy_expert.feature_columns

    if val_fs is not None:
        X_train, y_train, w_train = X_full, fs.y, weights
        X_val_full = dummy_expert._prepare_X(val_fs.X, val_fs.ranking_targets)
        X_val, y_val = X_val_full, val_fs.y
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(X_full, fs.y))
        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
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
