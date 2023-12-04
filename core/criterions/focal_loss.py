

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification introduced in:
        Lin et al. Focal Loss for Dense Object Detection. ICCV 2017 in https://arxiv.org/abs/1708.02002.

    The gamma parameter controls the focus of the loss. When gamma is 0, the loss is equivalent to cross entropy.
    When gamma is larger than 0, the loss is more focused on hard examples.
    """

    def __init__(self, reduction='mean', alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()

        self.reduction = reduction
        self.focal_alpha = alpha
        self.focal_gamma = gamma


    def forward(self, inputs, targets):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.focal_alpha * (1-BCE_EXP)**self.focal_gamma * BCE
                       
        return focal_loss
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('FocalLoss')
        parser.add_argument('--focal_alpha', type=float, default=0.5)
        parser.add_argument('--focal_gamma', type=float, default=2)
        return parent_parser