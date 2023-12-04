
# %% 
import os
from pathlib import Path
import torch
import seaborn as sns
import cloudpickle

def save_pickle(data, filename):
    with open(filename, 'wb') as handle:
        cloudpickle.dump(data, handle)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        data = cloudpickle.load(handle)
    return data


ROOT_PROJECT = Path(os.path.abspath(__file__)).parents[3].resolve()
HIST_PATH = os.path.join(os.getcwd(), 'hist_estimation.pickle')


class WeightedMSE(torch.nn.Module):

    def __init__(self, targets=None, weighting_scheme_path=HIST_PATH, weight_alpha=1, weight_epsilon=0.1, mse_weight=1, **kwargs) -> None:
        """

        Weighted MSE criterion.
        If no weights exist (i.e., if hist_path is not valid), then they will be built from `targets` through inverse density estimation.
        Else, the weights will be the ones previously defined in `hist_path`

        Parameters
        ----------

        `targets` - torch.tensor:
            Target values to build weighted MSE

        `hist_path` - Path:
            If existent, previously computed weights

        `alpha` - float: 
            Weighting factor that tells the model how important the rarer samples are;
            The higher the alpha, the higher the rarer samples' importance in the model.

        `epsilon` - float:
            Base value for the dense loss weighting function;
            Essentially, no data point is weighted with less than epsilon. So it always has at least epsilon importance;
        """

        super(WeightedMSE, self).__init__()
         
        self.weight_alpha = weight_alpha
        self.weight_epsilon = weight_epsilon
        self.mse_weight= mse_weight
        self.relu = torch.nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if weighting_scheme_path is not None and os.path.exists(weighting_scheme_path):
            self.pik_name = weighting_scheme_path
            self.freqs, self.ranges = load_pickle(self.pik_name)
        elif targets is not None:
            print("calculating histogram estimation...")
            self.freqs, self.ranges = self.hist_frequency_estimation(torch.flatten(targets), plot=False)
            save_pickle((self.freqs, self.ranges), f"{os.path.join('.', 'hist_estimation.pickle')}")
        else:
            ValueError("No targets were provided to build the weighting scheme")
        self.freqs = self.freqs.to(self.device)
        self.ranges = self.ranges.to(self.device)

    
    def hist_frequency_estimation(self, y:torch.Tensor, hist_len=10, plot=False):
        """
        Performs a histogram frequency estimation with y;\n
        The values of y are aggregated into hist_len ranges, then the density of each range
        is calculated and normalized.\n
        This serves as a good alternative to KDE since y is univariate.

        Parameters
        ----------
        `y` - torch.Tensor:
            estimation targets; must be one dimensional with values between 0 and 1;
        
        `hist_len` - int:
            number of ranges to use when aggregating the values of y

        `plot` - bool:
            plots the calculated histogram and shows the true count for each range

        Returns
        -------
        `hist_count` - torch.Tensor:
            tensor with the frequency of each range
        
        `hist_range` - torch.Tensor:
            the employed ranges
        """

        hist_range = torch.linspace(0, 1, hist_len + 1, device=self.device)[:-1] # the probabilities go from 0 to 1, with hist_len bins
        y = y.to(self.device)
        # calculates which bin each value of y belongs to
        #       y are multiplied by hist_len in order to retrieve their decimal bin
        hist_idxs = (hist_len*y).to(torch.int)
        hist_count = torch.bincount(hist_idxs, minlength=hist_len) # counts the occurrence of value in hist_idxs

        if plot:
            print(f"Histogram Bin /\t Count")
            step = hist_range[1] - hist_range[0]
            for i in range(len(hist_range)):
                print(f"[{hist_range[i]:.3f}, {hist_range[i] + step:.3f}[ : {hist_count[i]}")
                
        return hist_count, hist_range

    def get_dens_target(self, y:torch.Tensor, calc_weights = False):
        """
        Returns the density of each value in y following the `hist_frequency_estimation` result.
        """

        if calc_weights:
            self.freqs, self.ranges = self.hist_frequency_estimation(y)

        # hist bin that each y belongs to
        hist_idx = torch.abs(torch.unsqueeze(y, -1) - self.ranges).argmin(dim=-1)

        for idx in range(len(self.freqs)):
            # we replace the idx with its corresponding frequency
            hist_idx[hist_idx == idx] = self.freqs[idx]

        freq_min, freq_max = torch.min(self.freqs), torch.max(self.freqs)
       
        target_dens = (hist_idx - freq_min) / (freq_max - freq_min)
        return target_dens

    def get_weight_target(self, y:torch.Tensor):
        """
        Returns the weight value for each value in y according to the performed
        `hist_frequency_estimation`
        """
        y = y.to(self.device)
        y_dens = self.get_dens_target(y)
        weights = torch.max(1 - self.weight_alpha*y_dens, torch.full_like(y_dens, self.weight_epsilon, device=self.device))

        assert weights.shape == y.shape

        return weights / torch.mean(weights) # weights should have a mean of 1 to not directly influence the learning rate

    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        exp_y_pred, exp_y_gt = torch.broadcast_tensors(y_pred, y_gt) ## ensures equal dims
        weights_y_gt = self.get_weight_target(exp_y_gt)

        return torch.mean(self.mse_weight * weights_y_gt * (exp_y_gt - exp_y_pred)**2) ## weight_function * squared error

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('WeightedMSE')
        parser.add_argument('--weight_alpha', type=float, default=1, help='Weighting factor that tells the model how important the rarer samples are; The higher the alpha, the higher the rarer samples\' importance in the model.')
        parser.add_argument('--weight_epsilon', type=float, default=0.01, help='Base value for the dense loss weighting function; Essentially, no data point is weighted with less than epsilon. So it always has at least epsilon importance;')
        parser.add_argument('--mse_weight', type=float, default=1, help='Weighting factor for the MSE loss')
        parser.add_argument('--hist_path', type=str, default=HIST_PATH, help='Path to a previously computed histogram estimation')
        return parent_parser

if __name__ == "__main__":

    target = torch.randint(0, 8, (10,), dtype=torch.int64) / 8
    print(target)

    w_mse = WeightedMSE(target, weighting_scheme_path=None)

