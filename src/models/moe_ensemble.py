"""Orchestrates experts + gating network for blended predictions."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import EXPERT_TYPES, FIRST_YEAR, LAST_YEAR, SKIP_YEARS
from src.features.pipeline import FeatureSet, build_features_for_split
from src.models.gating_network import GatingNetwork
from src.models.tree_expert import TreeExpert

logger = logging.getLogger(__name__)


class MOEEnsemble:
    """Mixture of Experts: 3 TreeExperts + GatingNetwork → blended predictions."""

    def __init__(self, expert_params: dict | None = None):
        """Initialize MOE with optional per-expert param overrides.

        Args:
            expert_params: Dict of {expert_type: params_dict} for overrides.
        """
        self.expert_params = expert_params or {}
        self.experts: dict[str, TreeExpert] = {}
        self.gating: GatingNetwork | None = None
        self.blend_calibrator: IsotonicRegression | None = None

    def train_experts(
        self,
        train_fs: FeatureSet,
        val_fs: FeatureSet | None = None,
    ) -> None:
        """Train all 3 TreeExperts.

        Args:
            train_fs: Training FeatureSet.
            val_fs: Optional validation FeatureSet for early stopping.
        """
        for expert_type in EXPERT_TYPES:
            params = self.expert_params.get(expert_type)
            expert = TreeExpert(expert_type=expert_type, params=params)
            expert.fit(train_fs, val_fs=val_fs)
            self.experts[expert_type] = expert
            logger.info("Trained expert: %s", expert_type)

    def get_expert_predictions(self, fs: FeatureSet) -> np.ndarray:
        """Get predictions from all experts.

        Args:
            fs: FeatureSet to predict on.

        Returns:
            (n_samples, n_experts) array of predictions.
        """
        if not self.experts:
            raise RuntimeError("Must train_experts() first")

        preds = []
        for expert_type in EXPERT_TYPES:
            p = self.experts[expert_type].predict_proba(fs)
            preds.append(p)
        return np.column_stack(preds)

    def _get_gating_features(self, fs: FeatureSet) -> np.ndarray:
        """Extract gating features from FeatureSet, one-hot encoding round.

        Raw round (1-6) is replaced with 6 binary indicators so the gating
        network can learn round-specific expert routing without needing to
        discover nonlinear thresholds on a single low-magnitude integer.

        Args:
            fs: FeatureSet.

        Returns:
            (n, n_gating_features) array.  With round one-hot: 9 columns.
        """
        gating_cols = fs.gating_features
        if not gating_cols:
            raise ValueError("No gating features found in FeatureSet")

        raw = fs.X[gating_cols].values.copy()

        if "round" in gating_cols:
            round_idx = gating_cols.index("round")
            rounds = raw[:, round_idx].astype(int)
            other = np.delete(raw, round_idx, axis=1)
            # One-hot encode rounds 1-6
            round_oh = np.zeros((len(rounds), 6), dtype=np.float32)
            for r in range(1, 7):
                round_oh[:, r - 1] = (rounds == r).astype(np.float32)
            return np.hstack([other, round_oh])

        return raw

    def train_gating(
        self,
        context_X: np.ndarray,
        expert_preds: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Train gating network on held-out expert predictions.

        Args:
            context_X: (n, n_gating_features) gating context.
            expert_preds: (n, n_experts) held-out expert predictions.
            y: (n,) binary labels.
        """
        input_dim = context_X.shape[1]
        self.gating = GatingNetwork(input_dim=input_dim, n_experts=len(EXPERT_TYPES))
        self.gating.fit(context_X, expert_preds, y)

        # Post-hoc isotonic calibration on inner-CV blended predictions
        weights = self.gating.predict_weights(context_X)
        blended = (weights * expert_preds).sum(axis=1)
        self.blend_calibrator = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip",
        )
        self.blend_calibrator.fit(blended, y)
        logger.info("Fitted blend isotonic calibrator on %d inner-CV predictions", len(y))

    def predict_proba(self, fs: FeatureSet) -> np.ndarray:
        """Blended prediction: P = w1*P1 + w2*P2 + w3*P3, then isotonic calibration.

        Args:
            fs: FeatureSet to predict on.

        Returns:
            1-D array of blended probabilities.
        """
        expert_preds = self.get_expert_predictions(fs)

        if self.gating is not None:
            context_X = self._get_gating_features(fs)
            weights = self.gating.predict_weights(context_X)
        else:
            # Uniform weights fallback
            weights = np.ones_like(expert_preds) / expert_preds.shape[1]

        blended = (weights * expert_preds).sum(axis=1)

        if self.blend_calibrator is not None:
            blended = self.blend_calibrator.predict(blended)

        return blended

    def predict_decomposed(self, fs: FeatureSet) -> pd.DataFrame:
        """Full decomposition of predictions.

        Args:
            fs: FeatureSet to predict on.

        Returns:
            DataFrame with columns: p_seed, p_eff, p_unc, w_seed, w_eff, w_unc,
            p_blend_raw, p_blend.
        """
        expert_preds = self.get_expert_predictions(fs)

        if self.gating is not None:
            context_X = self._get_gating_features(fs)
            weights = self.gating.predict_weights(context_X)
        else:
            weights = np.ones_like(expert_preds) / expert_preds.shape[1]

        blended_raw = (weights * expert_preds).sum(axis=1)
        blended = blended_raw.copy()
        if self.blend_calibrator is not None:
            blended = self.blend_calibrator.predict(blended)

        return pd.DataFrame({
            "p_seed": expert_preds[:, 0],
            "p_eff": expert_preds[:, 1],
            "p_unc": expert_preds[:, 2],
            "w_seed": weights[:, 0],
            "w_eff": weights[:, 1],
            "w_unc": weights[:, 2],
            "p_blend_raw": blended_raw,
            "p_blend": blended,
        }, index=fs.X.index)

    def generate_inner_cv_predictions(
        self,
        train_seasons: list[int],
        merged_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inner LOYO loop to produce held-out expert predictions for gating training.

        For each year V in train_seasons:
            - Train experts on train_seasons - {V}
            - Predict V
            - Accumulate (context, expert_preds, y)

        Args:
            train_seasons: List of seasons in the training pool.
            merged_df: Full merged DataFrame for feature building.

        Returns:
            (context_X, expert_preds, y) accumulated across inner folds.
        """
        all_context = []
        all_expert_preds = []
        all_y = []

        for val_year in train_seasons:
            inner_train = [s for s in train_seasons if s != val_year]
            inner_train_fs, inner_val_fs = build_features_for_split(
                inner_train, [val_year], merged_df=merged_df,
            )

            # Train temporary experts on inner training set
            inner_experts = {}
            for expert_type in EXPERT_TYPES:
                params = self.expert_params.get(expert_type)
                expert = TreeExpert(expert_type=expert_type, params=params)
                expert.fit(inner_train_fs)
                inner_experts[expert_type] = expert

            # Predict on inner validation set
            preds = []
            for expert_type in EXPERT_TYPES:
                p = inner_experts[expert_type].predict_proba(inner_val_fs)
                preds.append(p)
            expert_preds = np.column_stack(preds)

            # Extract gating features (with one-hot round encoding)
            context = self._get_gating_features(inner_val_fs)

            all_context.append(context)
            all_expert_preds.append(expert_preds)
            all_y.append(inner_val_fs.y.values)

            logger.info(
                "Inner CV fold: val_year=%d, n_val=%d", val_year, len(inner_val_fs.y),
            )

        return (
            np.vstack(all_context),
            np.vstack(all_expert_preds),
            np.concatenate(all_y),
        )

    def train_full_nested(
        self,
        train_fs: FeatureSet,
        train_seasons: list[int],
        merged_df: pd.DataFrame,
    ) -> None:
        """Full nested training protocol.

        1. Inner CV → generate held-out expert predictions
        2. Train gating on inner-loop predictions
        3. Train final experts on full training pool

        Args:
            train_fs: Full training FeatureSet.
            train_seasons: Seasons in training pool.
            merged_df: Full merged DataFrame.
        """
        logger.info("Starting full nested training with %d seasons", len(train_seasons))

        # Step 1: Inner CV for gating training data
        context_X, expert_preds, y = self.generate_inner_cv_predictions(
            train_seasons, merged_df,
        )
        logger.info(
            "Inner CV complete: %d predictions for gating training", len(y),
        )

        # Step 2: Train gating network
        self.train_gating(context_X, expert_preds, y)

        # Step 3: Train final experts on full training data
        self.train_experts(train_fs)

        logger.info("Full nested training complete")

    def save(self, dir_path: str | Path) -> None:
        """Save all components to a directory.

        Args:
            dir_path: Directory to save into (created if needed).
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        for expert_type, expert in self.experts.items():
            expert.save(dir_path / f"expert_{expert_type}.pkl")

        if self.gating is not None:
            self.gating.save(dir_path / "gating_network.pkl")

        if self.blend_calibrator is not None:
            with open(dir_path / "blend_calibrator.pkl", "wb") as f:
                pickle.dump(self.blend_calibrator, f)

        # Save metadata
        meta = {"expert_params": self.expert_params}
        with open(dir_path / "moe_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("Saved MOEEnsemble to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str | Path) -> "MOEEnsemble":
        """Load a saved MOEEnsemble.

        Args:
            dir_path: Directory containing saved components.

        Returns:
            Loaded MOEEnsemble.
        """
        dir_path = Path(dir_path)

        # Load metadata
        with open(dir_path / "moe_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        moe = cls(expert_params=meta.get("expert_params", {}))

        # Load experts
        for expert_type in EXPERT_TYPES:
            expert_path = dir_path / f"expert_{expert_type}.pkl"
            if expert_path.exists():
                moe.experts[expert_type] = TreeExpert.load(expert_path)

        # Load gating
        gating_path = dir_path / "gating_network.pkl"
        if gating_path.exists():
            moe.gating = GatingNetwork.load(gating_path)

        # Load blend calibrator
        cal_path = dir_path / "blend_calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                moe.blend_calibrator = pickle.load(f)

        logger.info("Loaded MOEEnsemble from %s", dir_path)
        return moe
