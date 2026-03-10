"""MultiTaskHistoricalRankLoss: BCE + lambda * (1 - SpearmanR) per expert."""

import torch
import torch.nn as nn

from src.config import RANKING_LOSS_LAMBDA


class MultiTaskRankingLoss(nn.Module):
    """Combined BCE + differentiable Spearman ranking loss.

    Loss = BCE(P, y) + lambda * (1 - SpearmanR(P, ranking_target))

    Uses torchsort.soft_rank for differentiable ranking.
    For uncertainty expert (expert_idx=2): Spearman computed on |P - 0.5|
    vs certainty score (higher certainty = further from 0.5).
    """

    def __init__(
        self,
        ranking_lambda: float = RANKING_LOSS_LAMBDA,
        expert_idx: int = 0,
    ):
        super().__init__()
        self.ranking_lambda = ranking_lambda
        self.expert_idx = expert_idx
        self.bce = nn.BCELoss()

    def _soft_spearman(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Differentiable Spearman correlation via torchsort."""
        from torchsort import soft_rank

        pred_rank = soft_rank(pred.unsqueeze(0), regularization_strength=0.1).squeeze(0)
        target_rank = soft_rank(target.unsqueeze(0), regularization_strength=0.1).squeeze(0)

        pred_rank = pred_rank - pred_rank.mean()
        target_rank = target_rank - target_rank.mean()

        num = (pred_rank * target_rank).sum()
        den = torch.sqrt((pred_rank ** 2).sum() * (target_rank ** 2).sum() + 1e-8)
        return num / den

    def forward(
        self,
        pred: torch.Tensor,
        y_true: torch.Tensor,
        ranking_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: (n,) predicted probabilities.
            y_true: (n,) binary labels.
            ranking_target: (n,) ranking target values.

        Returns:
            Scalar loss.
        """
        bce_loss = self.bce(pred, y_true)

        # For uncertainty expert, use |P - 0.5| as the ranking quantity
        if self.expert_idx == 2:
            pred_for_rank = (pred - 0.5).abs()
        else:
            pred_for_rank = pred

        spearman_r = self._soft_spearman(pred_for_rank, ranking_target)
        rank_loss = 1.0 - spearman_r

        return bce_loss + self.ranking_lambda * rank_loss
