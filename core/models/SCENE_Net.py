from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.geneos.GENEO_kernel_torch import GENEO_kernel_torch
from core.models.geneos import cylinder, neg_sphere, arrow



def load_state_dict(model_path, gnet_class, model_tag='loss', kernel_size=None):
    """
    Returns SCENE-Net model and model_checkpoint
    """
    # print(model_path)
    # --- Load Best Model ---
    if os.path.exists(model_path):
        run_state_dict = torch.load(model_path)
        if model_tag == 'loss' and 'best_loss' in run_state_dict['models']:
            model_tag = 'best_loss'
        if model_tag in run_state_dict['models']:
            if kernel_size is None:
                kernel_size = run_state_dict['model_props'].get('kernel_size', (9, 6, 6))
            gnet = gnet_class(run_state_dict['model_props']['geneos_used'], 
                              kernel_size=kernel_size, 
                              plot=False)
            print(f"Loading Model in {model_path}")
            model_chkp = run_state_dict['models'][model_tag]

            try:
                gnet.load_state_dict(model_chkp['model_state_dict'])
            except RuntimeError: 
                for key in list(model_chkp['model_state_dict'].keys()):
                    model_chkp['model_state_dict'][key.replace('phi', 'lambda')] = model_chkp['model_state_dict'].pop(key) 
                gnet.load_state_dict(model_chkp['model_state_dict'])
            return gnet, model_chkp
        else:
            ValueError(f"{model_tag} is not a valid key; run_state_dict contains: {run_state_dict['models'].keys()}")
    else:
        ValueError(f"Invalid model path: {model_path}")

    return None, None


###############################################################
#                         GENEO Layer                         #
###############################################################

class GENEO_Layer(nn.Module):

    def __init__(self, geneo_class:GENEO_kernel_torch, kernel_size:tuple=None, smart=False):
        super(GENEO_Layer, self).__init__()  

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
            t_param = torch.tensor(config['geneo_params'][param], dtype=torch.float)
            t_param = nn.Parameter(t_param, requires_grad = not param in config['non_trainable'])
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
        kernel = geneo.kernel.to(dtype=torch.double)
        return kernel.view(1, *kernel.shape)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        geneo = self.geneo_class(self.name, self.kernel_size, plot=self.plot, **self.geneo_params)

        kernel = geneo.kernel.to(self.device, dtype=torch.double)
        
        return F.conv3d(x, kernel.view(1, 1, *kernel.shape), padding='same')



###############################################################
#                         SCENE-Nets                          #
###############################################################

class SCENE_Net(nn.Module):

    def __init__(self, geneo_num=None, kernel_size=None, plot=False,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Instantiates a SCENE-Net Module with specific GENEOs and their respective cvx coefficients.


        Parameters
        ----------
        `geneo_num` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize
        
        `kernel_size` - tuple/list:
            3 elem array with the kernel_size dimensions to discretize the GENEOs in (z, x, y) format

        `plot` - bool:
            if True plot information about the Module; It's propagated to submodules

        `device` - str:
            device where to load the Module.
        """
        super(SCENE_Net, self).__init__()

        self.device = device

        if geneo_num is None:
            self.sizes = {'cy'  : 1, 
                        'cone': 1, 
                        'neg' : 1}
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
        self.lambdas = (lambda_init_max - lambda_init_min)*torch.rand(num_lambdas, device=self.device, dtype=torch.float) + lambda_init_min
        self.lambdas = [nn.Parameter(lamb) for lamb in self.lambdas]
    
        self.lambda_names = [f'lambda_{key}_{i}' for key, val in self.sizes.items() for i in range(val)]
        self.last_lambda = self.lambda_names[torch.randint(0, num_lambdas, (1,))[0]]
        if plot:
            print(f"last cvx_coeff: {self.last_lambda}")

        # Updating last lambda
        self.lambdas_dict = dict(zip(self.lambda_names, self.lambdas)) # last cvx_coeff is equal to 1 - sum(lambda_i)
        self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
        
        self.lambdas_dict = nn.ParameterDict(self.lambdas_dict)

        if plot:
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

        conv_pred = torch.zeros_like(x)

        for i, g_name in enumerate(self.geneos):
            if f'lambda_{g_name}' == self.last_lambda:
                conv_pred += (1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda])*conv[:, [i]]
                #recompute last_lambda's actual value
                self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
            else:
                conv_pred += self.lambdas_dict[f'lambda_{g_name}']*conv[:, [i]]

        conv_pred = torch.relu(torch.tanh(conv_pred))

        return conv_pred


class SceneNet(nn.Module):

    def __init__(self, geneo_num=None, kernel_size=None, plot=False):
        """
        Instantiates a SCENE-Net Module with specific GENEOs and their respective cvx coefficients.

        Parameters
        ----------
        `geneo_num` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize
        
        `kernel_size` - tuple/list:
            3 elem array with the kernel_size dimensions to discretize the GENEOs in (z, x, y) format

        `plot` - bool:
            if True plot information about the Module; It's propagated to submodules

        `device` - str:
            device where to load the Module.
        """
        super(SceneNet, self).__init__()

        if geneo_num is None:
            self.sizes= {'cy' : 1, 
                        'cone': 1, 
                        'neg' : 1}
        else:
            self.sizes = geneo_num

        if kernel_size is not None:
            self.kernel_size = kernel_size
        # else is the default on GENEO_kernel_torch class

        self.geneos:Mapping[str, GENEO_Layer] = nn.ModuleDict()

        for key in self.sizes:
            if key == 'cy':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(cylinder.cylinderv2, kernel_size=kernel_size)

            elif key == 'cone':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(arrow.arrow, kernel_size=kernel_size)

            elif key == 'neg':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(neg_sphere.negSpherev2, kernel_size=kernel_size)

        # --- Initializing Convex Coefficients ---
        num_lambdas = sum(self.sizes.values())
        lambda_init_max = 1/num_lambdas
        lambda_init_min =  -2/num_lambdas # for testing purposes
        self.lambdas = (lambda_init_max - lambda_init_min)*torch.rand(num_lambdas, dtype=torch.float) + lambda_init_min
        self.lambdas = [nn.Parameter(lamb) for lamb in self.lambdas]
    
        self.lambda_names = [f'lambda_{key}_{i}' for key, val in self.sizes.items() for i in range(val)]
        self.last_lambda = self.lambda_names[torch.randint(0, num_lambdas, (1,))[0]]
        if plot:
            print(f"last cvx_coeff: {self.last_lambda}")

        # Updating last lambda
        self.lambdas_dict = dict(zip(self.lambda_names, self.lambdas)) # last cvx_coeff is equal to 1 - sum(lambda_i)
        self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
        
        self.lambdas_dict = nn.ParameterDict(self.lambdas_dict)

        if plot:
            print(f"Total Number of train params = {self.get_num_total_params()}")


    def get_cvx_coefficients(self):
        return self.lambdas_dict

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_parameters(self, detach=False):
        if detach:
            return {name: torch.tensor(param.detach()) for name, param in self.named_parameters()}
        return {name: param for name, param in self.named_parameters()}

    def get_geneo_params(self):
        return nn.ParameterDict(dict([(name.replace('.', '_'), p) for name, p in self.named_parameters() if not 'lambda' in name]))

    def get_model_parameters_in_dict(self):
        ddd = {}
        for key, val in self.named_parameters(): #update theta to remove the module prefix
            key_split = key.split('.')
            parameter_name = f"{key_split[-3]}.{key_split[-1]}" if 'geneo' in key else key_split[-1]
            ddd[parameter_name] = val.data.item()
        return ddd
        #return dict([(n, param.data.item()) for n, param in self.named_parameters()])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        kernels = torch.stack([self.geneos[geneo].compute_kernel() for geneo in self.geneos])
        conv = F.conv3d(x, kernels, padding='same')

        conv_pred = torch.zeros_like(x)

        for i, g_name in enumerate(self.geneos):
            if f'lambda_{g_name}' == self.last_lambda:
                conv_pred += (1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda])*conv[:, [i]]
                #recompute last_lambda's actual value
                self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
            else:
                conv_pred += self.lambdas_dict[f'lambda_{g_name}']*conv[:, [i]]

        conv_pred = torch.relu(torch.tanh(conv_pred))

        return conv_pred




###############################################################
#                     SCENE-Net Quantile                      #
###############################################################
class SCENENetQuantile(nn.Module):
    """
    Quantile Regressor powered by SCENE-Net architecture.
    Since SCENE-Net is domain specific, we create an ensemble of SCENE-Nets, one per quantile.
    """

    def __init__(self, geneo_num=None, kernel_size=None, qs=torch.tensor([0.1, 0.5, 0.9]), plot=False, model_path=None,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        """
        Instantiates a Quantile SCENE-Net Module with specific GENEOs and their respective cvx coefficients.


        Parameters
        ----------
        `geneo_num` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize
        
        `kernel_size` - tuple/list:
            3 elem array with the kernel_size dimensions to discretize the GENEOs in (z, x, y) format

        `qs` - torch.tensor:
            target quantiles to approximate

        `plot` - bool:
            if True plot information about the Module; It's propagated to submodules
        
        `model_path` - Path/str:
            if .pt file exists, it loads the existing model into this instance

        `device` - str:
            device where to load the Module.
        """
        super(SCENENetQuantile, self).__init__()

        if model_path is not None:
            # loads SCENE-Nets from a pre-existing model
            self.scnets = nn.ModuleList([load_state_dict(model_path, SCENE_Net, 'FBetaScore', kernel_size)[0] 
                                            for _ in range(len(qs))]).to(device)
        else:
            self.scnets = nn.ModuleList([SCENE_Net(geneo_num, kernel_size, plot) for _ in range(len(qs))]).to(device)

        self.qs = qs  
        self.device = device

        print(f"Total Number of train. params = {self.get_num_total_params()}")
        

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_dict_parameters(self):
        return dict([(n, param.data.item()) for n, param in self.named_parameters()])

    def get_cvx_coefficients(self):
        return [scnet.get_cvx_coefficients() for scnet in self.scnets]
    
    def get_geneo_params(self):
        return [scnet.get_geneo_params() for scnet in self.scnets]

    def forward(self, x:torch.Tensor):
        x_shape = x.shape # [B, 1, *vxg_size]
        pred = torch.empty((x_shape[0], len(self.qs), *x_shape[2:]), device=self.device)

        for i, _ in enumerate(self.qs):
            # pred comes in shape = [B, 1, *vxg_size]
            pred_q = self.scnets[i].forward(x)
            pred[:, i] = torch.squeeze(pred_q, dim=1)

        return pred





class SCENE_Net_Class(nn.Module):

    def __init__(self, geneo_num=None, plot=True, gnet_requires_grad=True, gnet_model_path = None):
        super().__init__()

        if gnet_model_path is None:
            self.gnet = SCENE_Net(geneo_num, plot)
        else:
            if os.path.exists(gnet_model_path):
                chkp = torch.load(gnet_model_path)
                self.gnet = SCENE_Net(geneo_num=chkp['geneos'])
                print(f"Loading Model in {gnet_model_path}")
                self.gnet.load_state_dict(chkp['model_state_dict'])
            else:
                ValueError("GENEO Net model path does not exist")

        if not gnet_requires_grad:
            for param in self.gnet.parameters():
                param.requires_grad = False
        
        tau_min = 0.2
        tau_max = 0.6
        self.tau = nn.Parameter((tau_max - tau_min)*torch.rand(1, dtype=torch.float)[0])

        if plot:
            for name, p in self.named_parameters():
                print(f"{name}: {p.item():.3f}, trainable:{p.requires_grad}, isleaf:{p.is_leaf}")
    
    
    def get_threshold(self):
        return self.tau

    def get_geneo_nums(self):
        return self.gnet.sizes

    def get_cvx_coefficients(self):
        return self.gnet.lambdas_dict

    def get_geneo_params(self):
        return nn.ParameterDict(dict([(name.replace('.', '_'), p) for name, p in self.gnet.named_parameters() if not 'lambda' in name]))

    def get_dict_parameters(self):
        return dict([(n, param.data.item()) for n, param in self.gnet.named_parameters()])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return (self.gnet(x) >= self.tau).to(x.dtype)

def main():
    gnet = SCENE_Net()
    for name, param in gnet.named_parameters():
        print(f"{name}: {type(param)}; {param}")
    return

if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    from core.datasets.ts40k import ToFullDense, torch_TS40Kv2, ToTensor
    from torchvision.transforms import Compose
    from criterions.quant_loss import QuantileGENEOLoss
    from criterions.w_mse import HIST_PATH 
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from utils.observer_utils import forward, process_batch, init_metrics, visualize_batch
    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity

    EXT_PATH = "/media/didi/TOSHIBA EXT/"
    TS40K_PATH = os.path.join(EXT_PATH, 'TS40K/')

    MODEL_PATH = "/home/didi/VSCode/soa_scenenet/scenenet_pipeline/torch_geneo/saved_scnets/models_geneo/2022-08-04 16:29:24.530075/gnet.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnet = SCENENetQuantile(None, (9, 6, 6), model_path=MODEL_PATH)

    for name, param in gnet.named_parameters():
        print(f"{name} = {param.item():.4f}")
    print("\n")

    input("?")


    composed = Compose([ToTensor(), ToFullDense()])
    ts40k = torch_TS40Kv2(dataset_path=TS40K_PATH, split='test', transform=composed)

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = QuantileGENEOLoss(torch.tensor([]), hist_path=HIST_PATH, alpha=1, rho=3, epsilon=0.1)
    tau=0.65
    test_metrics = init_metrics(tau) 
    test_loss = 0
    composed = Compose([ToTensor(), ToFullDense()])
    ts40k = torch_TS40Kv2(dataset_path=TS40K_PATH, split='test', transform=composed)

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = QuantileGENEOLoss(torch.tensor([]), hist_path=HIST_PATH, alpha=1, rho=3, epsilon=0.1)
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


    








