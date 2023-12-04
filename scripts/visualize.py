
import wandb
import os
import sys
import ast

# Vanilla PyTorch
import torch
from torchvision.transforms import Compose

# PyTorch Lightning
import pytorch_lightning as pl

# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from constants import ROOT_PROJECT, TS40K_PATH, WEIGHT_SCHEME_PATH, get_experiment_config_path, get_experiment_path

import utils.pcd_processing as eda
import utils.scripts_utils as su

import core.lit_modules.lit_callbacks as lit_callbacks
import core.lit_modules.lit_model_wrappers as lit_models
from core.lit_modules.lit_data_wrappers import LitTS40K

from core.datasets.torch_transforms import Voxelization, ToTensor, ToFullDense


def init_model(ckpt_path):

    if wandb.config.load_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = lit_models.LitSceneNet.load_from_checkpoint(ckpt_path)
    else:
        # Model random initialization
        geneo_config = {
            'cy'   : wandb.config.cylinder_geneo,
            'cone' : wandb.config.arrow_geneo,
            'neg'  : wandb.config.neg_sphere_geneo, 
        }

        model = lit_models.LitSceneNet(geneo_config,
                                       ast.literal_eval(wandb.config.kernel_size))
        
    return model


def init_data(data_path):
    vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
    vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size, vox_size=vox_size),
                        ToTensor(), 
                        ToFullDense(apply=(True, True))])

    data_module = LitTS40K(data_path,
                           wandb.config.batch_size,
                           composed,
                           wandb.config.num_workers,
                           wandb.config.val_split,
                           wandb.config.test_split)
    
    return data_module


def main():


    # ------------------------
    # 3 INIT MODEL
    # ------------------------

    ckpt_dir = wandb.config.checkpoint_dir
    ckpt_path = os.path.join(ckpt_dir, wandb.config.checkpoint_metric + '.ckpt')

    model = init_model(None, ckpt_path)

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    data_path = wandb.config.data_path

    if not os.path.exists(wandb.config.data_path):
        data_path = TS40K_PATH
        wandb.config.update({'data_path': data_path}, allow_val_change=True) # override data path

    data_module = init_data(data_path)

    
    # ------------------------

    trainer = pl.Trainer()

    trainer.predict(model, data_module)



if __name__ == '__main__':

    print(f"{'='*20} Visualization {'='*20}")