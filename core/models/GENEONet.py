from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import sys

from scenenet_pipeline.torch_geneo.geneos import arrow

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from VoxGENEO import Voxelization as Vox
from torch_geneo.geneos import GENEO_kernel_torch, cylinder, neg_sphere


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# GENEO-Layer

class GENEO_Layer(nn.Module):

    def __init__(self, geneo_class:GENEO_kernel_torch.GENEO_kernel_torch, kernel_size:tuple=None, smart=False):
        super().__init__()  

        self.geneo_class = geneo_class

        self.init_from_config(smart)

        if kernel_size is not None:
            self.kernel_size = kernel_size

    
    def init_from_config(self, smart=False):

        if smart:
            config = self.geneo_class.geneo_smart_config()
            if config['plot']:
                print("JSON file GENEO Initialization...")
        else:
            config = self.geneo_class.geneo_random_config()
            if config['plot']:
                print("Random GENEO Initialization...")


        self.name = config['name']
        self.kernel_size = config['kernel_size']
        self.plot = config['plot']
        self.geneo_params = {}

        for param in config['geneo_params']:
            t_param = nn.Parameter(config['geneo_params'][param].to(torch.float), requires_grad= not param in config['non_trainable'])
            self.geneo_params[param] = t_param

        self.geneo_params = nn.ParameterDict(self.geneo_params)


    def init_from_kwargs(self, kernel_size, kwargs):
        self.kernel_size = kernel_size
        self.geneo_params = {}
        self.name = 'GENEO'
        self.plot = False
        for param in self.geneo_class.mandatory_parameters():
            
            self.geneo_params[param] = nn.Parameter(torch.tensor(kwargs[param], dtype=torch.float))

        self.geneo_params = nn.ParameterDict(self.geneo_params)

    def compute_kernel(self) -> torch.Tensor:
        geneo = self.geneo_class(self.name, self.kernel_size, plot=self.plot, **self.geneo_params)
        kernel = geneo.kernel.to(device, dtype=torch.double)
        return kernel.view(1, *kernel.shape)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        geneo = self.geneo_class(self.name, self.kernel_size, plot=self.plot, **self.geneo_params)

        kernel = geneo.kernel.to(device, dtype=torch.double)
        
        return F.conv3d(x, kernel.view(1, 1, *kernel.shape), padding='same')



# SCENE-Net

class GENEONet(nn.Module):

    def __init__(self, geneo_num=None, kernel_size=None, plot=False):
        super().__init__()

        if geneo_num is None:

            self.sizes = {'cy':1, 
                          'cone': 1, 
                          'neg': 1}
        else:
            self.sizes = geneo_num

        if kernel_size is not None:
            self.kernel_size = kernel_size
        # else is the default on GENEO_kernel_torch class

        self.geneos:Mapping[str, GENEO_Layer] = nn.ModuleDict()

        for key in self.sizes:
            if key == 'cy':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(cylinder.cylinder_kernel, kernel_size=kernel_size)

            elif key == 'cone':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(arrow.cone_kernel, kernel_size=kernel_size)

            elif key == 'neg':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(neg_sphere.neg_sphere_kernel, kernel_size=kernel_size)

        # --- Initializing Convex Coefficients ---
        num_lambdas = sum(self.sizes.values())
        lambda_init_max = 0.6 #2 * 1/num_lambdas
        lambda_init_min =  0 #-1/num_lambdas # for testing purposes
        self.lambdas = (lambda_init_max - lambda_init_min)*torch.rand(num_lambdas, device=device, dtype=torch.float) + lambda_init_min
        self.lambdas = [nn.Parameter(lamb) for lamb in self.lambdas]
       
        self.lambda_names = [f'lambda_{key}_{i}' for key, val in self.sizes.items() for i in range(val)]
        self.last_lambda = self.lambda_names[torch.randint(0, num_lambdas, (1,))[0]]
        if plot:
            print(f"last cvx_coeff: {self.last_lambda}")

        #updating last lambda
        self.lambdas_dict = dict(zip(self.lambda_names, self.lambdas)) # last cvx_coeff is equal to 1 - sum(lambda_i)
        self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
        # print(self.lambda_names)
        # assert len(self.lambdas) == len(self.lambda_names) == len(self.geneos)
        # assert np.isclose(sum(self.lambdas_dict.values()).data.item(), 1), sum(self.lambdas_dict.values()).data.item()
        self.lambdas_dict = nn.ParameterDict(self.lambdas_dict)

        print(f"Total Number of train. params = {self.get_num_total_params()}")

    def get_geneo_nums(self):
        return self.sizes

    def get_cvx_coefficients(self):
        return self.lambdas_dict

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_geneo_params(self):
        return nn.ParameterDict(dict([(name.replace('.', '_'), p) for name, p in self.named_parameters() if not 'lambda' in name]))

    def get_dict_parameters(self):
        return dict([(n, param.data.item()) for n, param in self.named_parameters()])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        kernels = torch.stack([self.geneos[geneo].compute_kernel() for geneo in self.geneos])
        conv = F.conv3d(x, kernels, padding='same')

        # print(kernels.shape)
        # print(x.shape)
        # print(conv.shape)
        # print(conv[:, [0]].shape)

        conv_pred = torch.zeros_like(x)

        for i, g_name in enumerate(self.geneos):
            if f'lambda_{g_name}' == self.last_lambda:
                conv_pred += (1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda])*conv[:, [i]]
                #recompute last_lambda's actual value
                self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
            else:
                conv_pred += self.lambdas_dict[f'lambda_{g_name}']*conv[:, [i]]

            # print(f"{g_name}: {self.lambdas_dict[f'lambda_{g_name}']}")
            # Vox.plot_voxelgrid((self.lambdas_dict[f'lambda_{g_name}']*conv[:, i][0]).cpu().detach().numpy(), title=f'{g_name} observation output')
        
        #print(self.lambdas_dict[self.last_lambda])

        #conv_pred = self.lambdas_dict['lambda_cy']*conv[:, 0] + self.lambdas_dict['lambda_cone']*conv[:, 1] + self.lambda_neg*conv[:, 2]
        # Vox.plot_voxelgrid(conv[0, 0].cpu().detach().numpy()) # cylinder convolution

        #print(conv_pred.shape)
        #Vox.plot_voxelgrid(conv_pred[0].cpu().detach().numpy())

        if False:
            #Vox.plot_voxelgrid(x[0][0].cpu().detach())

            print(f"min/max conv_pred values = {torch.min(conv_pred):.4f} ; {torch.max(conv_pred):.4f}")
            cvx_explore = conv_pred[0][0].cpu().detach()
            print(cvx_explore.shape)

            Vox.plot_voxelgrid(cvx_explore)
        
            # plt.hist(torch.flatten(conv_pred[conv_pred != 0]).cpu().detach().numpy())
            # plt.title("Distribution of values of Observer")
            # plt.show()
            # plt.hist(torch.flatten(torch.tanh(conv_pred[conv_pred != 0])).cpu().detach().numpy())
            # plt.title("Distribution of values of Observer + tanh")
            # plt.show()
            # plt.hist(torch.flatten(torch.sigmoid(conv_pred[conv_pred != 0])).cpu().detach().numpy())
            # plt.title("Distribution of values of Observer + sig")
            # plt.show()
            # plt.hist(torch.flatten(torch.relu(torch.tanh(conv_pred[conv_pred != 0]))).cpu().detach().numpy())
            # plt.title("Distribution of values of Observer + tanh + relu")
            # plt.show()
            # pred_tanh = torch.tanh(cvx_explore)
            # pred_sig = torch.sigmoid(cvx_explore)
            # #Vox.plot_voxelgrid(cvx_explore, color_mode='ranges')
            # Vox.plot_voxelgrid(torch.where(pred_tanh >= 0.7, pred_tanh, torch.tensor(0.0, dtype=torch.double)), color_mode='ranges')
            # Vox.plot_voxelgrid(torch.where(pred_sig >= 0.7, pred_sig, torch.tensor(0.0, dtype=torch.double)), color_mode='ranges')
            # Vox.plot_voxelgrid(torch.where((cvx_explore >= 0.8), cvx_explore, torch.tensor(0.0, dtype=torch.double)), color_mode='ranges')
            # Vox.plot_voxelgrid(torch.where((cvx_explore >= 0.0) & (cvx_explore <= 0.4) & (cvx_explore != 0), cvx_explore, torch.tensor(0.0, dtype=torch.double)), color_mode='ranges')
            # Vox.plot_voxelgrid(torch.where((cvx_explore >= 0.0) & (cvx_explore <= 0.4) & (cvx_explore != 0), cvx_explore, torch.tensor(0.0, dtype=torch.double)), color_mode='ranges')


        conv_pred = torch.relu(torch.tanh(conv_pred))

        #Vox.plot_voxelgrid(conv_pred[0][0].cpu().detach().numpy())

        return conv_pred

import os

def main():
    gnet = GENEONet()
    for name, param in gnet.named_parameters():
        print(f"{name}: {type(param)}; {param}")
    return

if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    from datasets.ts40k import ToFullDense, torch_TS40K, ToTensor
    from torchvision.transforms import Compose
    from scenenet_pipeline.torch_geneo.models.geneo_loss import HIST_PATH
    from scenenet_pipeline.torch_geneo.models.geneo_loss import GENEO_Loss
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from utils.observer_utils import forward, process_batch, init_metrics, visualize_batch
    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity


    ROOT_PROJECT = str(Path(os.getcwd()).parent.absolute().parent.absolute())
    print(ROOT_PROJECT)
    SAVE_DIR = ROOT_PROJECT + "/dataset/torch_dataset"

    gnet = GENEONet(None, (9, 6, 6)).to(device)

    composed = Compose([ToTensor(), ToFullDense()])
    ts40k = torch_TS40K(dataset_path=SAVE_DIR, split='test', transform=composed)

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = GENEO_Loss(torch.tensor([]), hist_path=HIST_PATH, alpha=1, rho=3, epsilon=0.1)
    tau=0.65
    test_metrics = init_metrics(tau) 
    test_loss = 0
 
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                vox, gt = ts40k[0]
                gnet(vox[None])

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


        input("\n\nContinue?\n\n")


        for batch in tqdm(ts40k_loader, desc=f"testing..."):
            loss, pred = process_batch(gnet, batch, geneo_loss, None, test_metrics, requires_grad=False)
            test_loss += loss

            test_res = test_metrics.compute()
            pre = test_res['Precision']
            rec = test_res['Recall']

            # print(f"Precision = {pre}")
            # print(f"Recall = {rec}")
            #if pre <= 0.1 or rec <= 0.1:
            if pre >= 0.3 and rec >= 0.20:
            #if True:
                print(f"Precision = {pre}")
                print(f"Recall = {rec}")
                vox, gt = batch
                visualize_batch(vox, gt, pred, tau=tau)
                input("\n\nPress Enter to continue...")

            test_metrics.reset()

        test_loss = test_loss /  len(ts40k_loader)
        test_res = test_metrics.compute()
        print(f"\ttest_loss = {test_loss:.3f};")
        for met in test_res:
            print(f"\t{met} = {test_res[met]:.3f};")


    








