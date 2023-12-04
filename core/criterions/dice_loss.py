

import torch
import torch.nn as nn
import torch.nn.functional as F


from core.criterions.w_mse import WeightedMSE

class BinaryDiceLoss(nn.Module):
    """
    Dice loss of binary class

    Parameters
    ----------
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns
    -------
        Loss tensor according to arg reduction
    Raise
    -----
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.power = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.power) + target.pow(self.power), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



class BinaryDiceLoss_BCE(WeightedMSE):
    """
    Weighted Binary Dice Loss + BCE Loss
    """

    def __init__(self, targets=..., weighting_scheme_path=..., weight_alpha=1, weight_epsilon=0.1, mse_weight=1, reduction='mean', **kwargs) -> None:
        super().__init__(targets, weighting_scheme_path, weight_alpha, weight_epsilon, mse_weight, **kwargs)

        self.dice = BinaryDiceLoss(reduction=reduction)
        self.bce = nn.BCELoss(reduction='none')

        self.reduction = reduction

    def forward(self, predict, target):
        """
        Parameters
        ----------
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        """

        weights = self.get_weight_target(target)

        bce_loss = self.bce(predict, target)
        dice_loss = self.dice(predict, target)

    
        if self.reduction == 'mean':
            return torch.mean(weights*bce_loss) + dice_loss
        elif self.reduction == 'sum':
            return torch.sum(weights*bce_loss) + dice_loss
        elif self.reduction == 'none':
            return weights*bce_loss + dice_loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

        

    @staticmethod
    def add_model_specific_args(parent_parser):
        return super().add_model_specific_args(parent_parser)



   
