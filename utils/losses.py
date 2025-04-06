import torch
from typing import Union
from neuralforecast.losses.pytorch import BasePointLoss


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Safely divide two tensors, replacing NaN and infinite values with zero."""
    div = a / b
    return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)


def _weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute the weighted mean of individual losses."""
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))


class customLoss(BasePointLoss):
    """Custom loss function that computes a normalized weighted mean squared error."""

    def __init__(self, horizon_weight: Union[torch.Tensor, None] = None):
        super(customLoss, self).__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,
            output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor,
        mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """Compute the custom loss value."""
        denominator = y.clone()
        denominator[denominator == 0] = 1.0

        losses = ((y - y_hat) ** 2) / denominator
        weights = self._compute_weights(y=y, mask=mask)

        return _weighted_mean(losses=losses, weights=weights)
