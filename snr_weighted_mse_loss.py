import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, yhat, y, weights):
        """
        Compute the weighted MSE loss.

        Args:
            yhat (torch.Tensor): Predicted values (batch_size, ...)
            y (torch.Tensor): Target values (batch_size, ...)
            weights (torch.Tensor): Weights for each sample in the batch (batch_size,)

        Returns:
            torch.Tensor: Weighted MSE loss.
        """
        # Ensure weights shape matches batch size
        if weights.dim() == 1:
            weights = weights.view(-1, *[1] * (yhat.dim() - 1))  # Expand dimensions if needed

        mse = (yhat - y) ** 2  # Element-wise squared error
        weighted_mse = weights * mse  # Apply weights to the squared error
        return weighted_mse.mean()  # Mean over all elements
