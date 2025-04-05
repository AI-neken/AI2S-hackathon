import torch
from typing import Union
from neuralforecast.losses.pytorch import BasePointLoss


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)

def _weighted_mean(losses, weights):
    """
    Compute weighted mean of losses per datapoint.
    """
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))

class customLoss(BasePointLoss):

    def __init__(self, horizon_weight=None):
        super(customLoss, self).__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        y_hat: tensor, Predicted values.<br>
        mask: tensor, Specifies datapoints to consider in loss.<br>

        **Returns:**<br>
        mse: tensor (single value).
        """
        denominator = y.clone()
        denominator[denominator == 0] = 1
        losses = ((y - y_hat) ** 2) / denominator
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)
        