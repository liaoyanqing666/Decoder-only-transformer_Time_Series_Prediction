import torch
import torch.nn as nn

class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_true, y_pred):
        """
        Calculate SMAPE Loss

        Parameters:
            - y_true: True values
            - y_pred: Predicted values

        Returns:
            - smape_loss: SMAPE Loss
        """
        numerator = torch.abs(y_true - y_pred)
        denominator = torch.abs(y_true) + torch.abs(y_pred)
        smape = 2 * torch.mean(torch.div(numerator, denominator + 1e-8))  # Adding a small epsilon to avoid division by zero
        return smape
