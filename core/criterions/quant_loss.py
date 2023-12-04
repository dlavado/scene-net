







from typing import List
import torch

import sys
from pathlib import Path

sys.path.insert(0, '..')
sys.path.insert(1, '../../..')
from scenenet_pipeline.torch_geneo.criterions.w_mse import WeightedMSE, HIST_PATH
from scenenet_pipeline.torch_geneo.criterions.geneo_loss import GENEO_Loss


class QuantileLoss(WeightedMSE):


    def __init__(self, targets: torch.Tensor, qs=torch.tensor([0.1, 0.5, 0.9]), 
                 hist_path=HIST_PATH, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
        """
        Weighted Quantile Loss Based on WeightedMSE weighting scheme

        Parameters
        ----------
         `targets` - torch.tensor:
            Target values to build weighted MSE

        `qs` - torch.tensor:
            target quantiles to approximate

        `hist_path` - Path:
            If existent, previously computed weights

        `alpha` - float: 
            Weighting factor that tells the model how important the rarer samples are;
            The higher the alpha, the higher the rarer samples' importance in the model.

        `rho` - float: 
            Regularizing term that punishes negative convex coefficients;

        `epsilon` - float:
            Base value for the dense loss weighting function;
            Essentially, no data point is weighted with less than epsilon. So it always has at least epsilon importance;
        """

        super().__init__(targets, hist_path, alpha, rho, epsilon, gamma)


        assert torch.all(qs < 1) and torch.all(qs > 0) # quantiles in admissible range

        self.qs = qs.reshape(-1, 1).to(self.device)
        self._qs = torch.concat((self.qs, self.qs - 1), dim=-1).to(self.device) # expanding quantiles with `qs - 1` (inverse) to fasten computations

    def transform_forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        return y_pred.to(self.device), y_gt.to(self.device)


    def data_fidelity(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        return (y_gt - y_pred).to(self.device)


    def deprecated(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        d_fid = self.data_fidelity(y_pred, y_gt)
        d_fid = torch.unsqueeze(d_fid, dim=-1) #add dim to multiply by quantiles

        return torch.mean(torch.max(self._qs*d_fid, dim=-1)[0], dim=-1)

    def quantile_loss(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        """
        Returns the average Quantile loss for each sample in Batch dim
        """

        d_fid = self.data_fidelity(y_pred, y_gt)
        #d_fid = torch.unsqueeze(d_fid, dim=-1) #add dim to multiply by quantiles

        q_loss = []
        #q_loss is a list of the Quantile Loss of each quantile, thus, each elem is a vector with the q_loss for each sample
        for i, q in enumerate(self.qs):
            d_fid = self.data_fidelity(y_pred[:, i], y_gt)
            q_loss.append(torch.max(q*d_fid, (q - 1)*d_fid).unsqueeze(dim=1))

        loss = torch.sum(torch.cat(q_loss, dim=1), dim=1)
        return loss


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        """
        Returns a weighted Quantile loss following the weighting scheme defined in WeightedMSE
        """

        y_pred, y_gt = self.transform_forward(y_pred, y_gt)
        weights_y_gt = self.get_weight_target(y_gt)

        q_loss = self.quantile_loss(y_pred, y_gt)

        return torch.mean(weights_y_gt*q_loss)


class QuantileGENEOLoss(QuantileLoss, GENEO_Loss):

    def __init__(self, targets: torch.Tensor, qs=torch.tensor([0.1, 0.5, 0.9]), hist_path=HIST_PATH, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
       
       super(QuantileGENEOLoss, self).__init__(targets, qs, hist_path, alpha, rho, epsilon, gamma)


    def cvx_loss(self, cvx_coeffs:List[torch.nn.ParameterDict]):
        return sum(super(QuantileGENEOLoss, self).cvx_loss(cvx_c) for cvx_c in cvx_coeffs)

    def positive_regularizer(self, params: List[torch.nn.ParameterDict]):
        return sum(super(QuantileGENEOLoss, self).positive_regularizer(g_params) for g_params in params)

    def forward(self,  y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:List[torch.nn.ParameterDict], geneo_params:List[torch.nn.ParameterDict]):
        """
        Parameters
        ----------

        `y_pred` - torch.Tensor:
            Module Prediction

        `y_gt` - torch.Tensor:
            Ground Truth

        `cvx_coeffs` - List[torch.nn.ParameterDict]:
            List with each SCENE-Net's cvx_coeffs

        `geneo_params` - List[torch.nn.ParameterDict]:
            List with each SCENE-Net's GENEO Parameters
        """

        data_fidelity = QuantileLoss.forward(self, y_pred, y_gt)

        non_cvx_penalty = self.cvx_loss(cvx_coeffs)

        neg_penalty = self.positive_regularizer(geneo_params)

        return data_fidelity + non_cvx_penalty + neg_penalty



if __name__ == '__main__':

    NUM_SAMPLES = 1024
    B_SIZE = 4

    qs = torch.tensor([0.1, 0.5, 0.9])

    y_pred = torch.rand((NUM_SAMPLES, len(qs), 8, 8, 8))
    y_true = torch.rand(NUM_SAMPLES, 1, 8, 8, 8)

    q_loss = QuantileLoss(y_true, hist_path=None)

    for i in torch.randint(0, NUM_SAMPLES-B_SIZE, size=(5,)):

        pred = y_pred[i:i+B_SIZE]
        target = y_true[i:i+B_SIZE]

        # print(target)
        # print()
        # print(q_loss.data_fidelity(pred, target))
        # print()
        print(q_loss(pred, target))
        #print(torch.mean(q_loss.quantile_loss(pred, target)))

        input("Continue?")




    


