# %%
import itertools
import time
from matplotlib import cm
import numpy as np
import sympy as smp
import sympy.vector as smpv
import sympy.physics.vector as spv
import sympytorch as spt
from scipy import integrate as intg
from sklearn.metrics import precision_score, recall_score, f1_score

import IPython.display as disp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import torch


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
from scripts import constants as const
from utils import voxelization as Vox
from utils import pcd_processing as eda
from core.models.geneos.GENEO_kernel_torch import GENEO_kernel_torch


class cylinder_kernel(GENEO_kernel_torch):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        """
        Creates a 3D torch tensor that demonstrates a cylinder.\n

        Parameters
        ----------
        radius - float:
        radius of the cylinder's base; radius <= kernel_size[1];

        sigma - float:
        variance for the gaussian function when assigning weights to the kernel;

        Returns
        -------
            3D torch tensor with the cylinder kernel 
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")

        self.radius = kwargs['radius'].to(self.device)

        if plot:
            print("--- Cylinder Kernel ---")
            print(f"radius = {self.radius:.4f}; {type(self.radius)};")

        self.sigma = 1
        if kwargs.get('sigma') is not None:
            self.sigma = kwargs['sigma']
            if plot:
                print(f"sigma = {self.sigma:.4f}; {type(self.sigma)};")           

        self.plot = plot
        
        super().__init__(name, kernel_size)  


    def gaussian(self, x:torch.Tensor, epsilon=0) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        circle_x = x_c_norm**2 - (self.radius + epsilon)**2 

        return torch.exp((circle_x**2) * (-1 / (2*self.sigma**2)))

    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size[1:])) 
 
    def compute_kernel(self, plot=False):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2)
        
        floor_vals = self.gaussian(floor_idxs)
        
        floor_vals = self.sum_zero(floor_vals)
        floor_vals = torch.t(floor_vals).view(self.kernel_size[1:])            
        #assert floor_vals.requires_grad
    
        kernel = torch.tile(floor_vals, (self.kernel_size[0], 1, 1))
        # assert kernel.shape == self.kernel_size
        # assert kernel.requires_grad
        # assert torch.equal(kernel[0], floor_vals)
        # assert torch.sum(kernel) <= 1e-10 or torch.sum(kernel) <= -1e-10 # weight sum == 0

        return kernel

       
    def mandatory_parameters():
        return ['radius']
    
    def geneo_parameters():
        return cylinder_kernel.mandatory_parameters() + ['sigma']

    def geneo_random_config(name='GENEO_rand'):
        rand_config = GENEO_kernel_torch.geneo_random_config()

        geneo_params = {
            'radius' : torch.randint(1, rand_config['kernel_size'][1], (1,))[0] / 2 ,
            'sigma' : torch.randint(5, 10, (1,))[0] / 5 #float \in [1, 2]
        }   

        rand_config['geneo_params'] = geneo_params
        rand_config['name'] = 'cylinder'

        return rand_config

    def geneo_smart_config(name="Smart_Cylinder"):

        config = {
            'name' : name,
            'kernel_size': (9, 6, 6),
            'plot': False,
            'non_trainable' : [],

            'geneo_params' : {
                                'radius' :  torch.tensor(1.0) ,
                                'sigma' :  torch.tensor(2.0)
                             }
        }


        return config





class cylinderv2(cylinder_kernel):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        super().__init__(name, kernel_size, plot, **kwargs)


    def gaussian(self, x:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 #- (self.radius + epsilon)**2 

        return self.sigma*torch.exp((gauss_dist**2) * (-1 / (2*(self.radius + epsilon)**2)))


    def compute_kernel(self, plot=False):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2)

        floor_vals = self.gaussian(floor_idxs)
        floor_vals = self.sum_zero(floor_vals)
        floor_vals = torch.t(floor_vals).view(self.kernel_size[1:])            
        #assert floor_vals.requires_grad
    
        kernel = torch.tile(floor_vals, (self.kernel_size[0], 1, 1))

        return kernel 



def plot_R2func(func, lim_x1, lim_x2, cmap=cm.coolwarm):
    x1_lin = np.linspace(lim_x1[0], lim_x1[1], 100)
    x2_lin = np.linspace(lim_x2[0], lim_x2[1], 100)
    x1_lin, x2_lin = np.meshgrid(x1_lin, x2_lin)
    g = func(x1_lin, x2_lin)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_lin, x2_lin, g, cmap=cmap)
    plt.show()

# %%
if __name__ == "__main__":
    from torchvision.transforms import Compose
    from core.datasets.torch_transforms import Voxelization, ToTensor, ToFullDense
    from core.datasets.ts40k import TS40K

    vxg_size = (64, 64, 64)
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size, vox_size=None),
                        ToTensor(), 
                        ToFullDense(apply=(True, True))])
    
    #ts40k = TS40K(dataset_path=const.TS40K_PATH, transform=composed)


    # vox, vox_gt = ts40k[2]
    # vox, vox_gt = vox.to(torch.float), vox_gt.to(torch.float)
    # print(vox.shape)
    # Vox.plot_voxelgrid(vox.numpy()[0])
    # Vox.plot_voxelgrid(vox_gt.numpy()[0])
    


    cy = cylinder_kernel('cy', (6, 6, 6), radius=torch.tensor(2), sigma=torch.tensor(2))
    cy = cylinderv2('cy', (6, 7, 7), radius=torch.tensor(2.5), sigma=torch.tensor(5))
    #kernel = cy.compute_kernel_(True)

    print(cy.kernel.shape)
    print(cy.kernel[0, :, :])

    cy.plot_kernel()

    #cy.convolution(vox.view((1, *vox.shape)).to(cy.device),plot=True)


    type(cy.kernel)

# %%
