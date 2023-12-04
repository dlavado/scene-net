
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.5
BETA = 1
GAMMA = 2


class TverskyLoss(nn.Module):
    """

    Tversky loss for binary classification introduced in: 
        Tversky loss function for image segmentation using 3D FCDN. 2017 in https://arxiv.org/abs/1706.05721.

    in the case of alpha=beta=0.5 the Tversky index simplifies to be the same as the Dice coefficient, which is also equal to the F1 score

    With alpha=beta=1, Equation 2 produces Tanimoto coefficient, and setting alpha+beta=1 produces the set of Fβ scores.
    Larger betas weigh recall higher than precision, and vice versa for smaller betas.

    Parameters:
    ----------
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        smooth: smooth factor to avoid division by zero.
    """


    def __init__(self, tversky_alpha=ALPHA, tversky_beta=BETA, tversky_smooth=1, **kwargs):
        super(TverskyLoss, self).__init__()

        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_smooth = tversky_smooth

    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.tversky_smooth) / (TP + self.tversky_alpha*FP + self.tversky_beta*FN + self.tversky_smooth)  
        
        return 1 - Tversky
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('TverskyLoss')
        parser.add_argument('--tversky_alpha', type=float, default=ALPHA)
        parser.add_argument('--tversky_beta', type=float, default=BETA)
        parser.add_argument('--tversky_smooth', type=float, default=1)
        return parent_parser


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for binary classification introduced in:
        Focal Tversky loss: a novel loss function and DSC (Dice score) maximization approach for lesion segmentation. 2019 in https://arxiv.org/abs/1810.07842.

    Parameters:
    ----------
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        gamma: controls the penalty for easy examples.
    """
    def __init__(self, tversky_alpha=ALPHA, tversky_beta=BETA, focal_gamma=GAMMA, tversky_smooth=1, **kwargs):
        super(FocalTverskyLoss, self).__init__()

        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_smooth = tversky_smooth
        self.focal_gamma = focal_gamma


    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.tversky_smooth) / (TP + self.tversky_alpha*FP + self.tversky_beta*FN + self.tversky_smooth)  
        FocalTversky = (1 - Tversky)**self.focal_gamma
                       
        return FocalTversky
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('FocalTverskyLoss')
        parser.add_argument('--tversky_alpha', type=float, default=ALPHA, help='controls the penalty for false positives')
        parser.add_argument('--tversky_beta', type=float, default=BETA, help='controls the penalty for false negatives')
        parser.add_argument('--tversky_smooth', type=float, default=1, help='smooth factor to avoid division by zero')
        parser.add_argument('--focal_gamma', type=float, default=GAMMA, help='controls the penalty for easy examples')
        return parent_parser