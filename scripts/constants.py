

import os
import sys
from pathlib import Path
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_experiment_path(model, dataset) -> Path:
    return os.path.join(get_project_root(), 'experiments', f"{model}_{dataset}")

def get_experiment_config_path(model, dataset) -> Path:
    return os.path.join(get_experiment_path(model, dataset), 'config.yml')


ROOT_PROJECT = get_project_root()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if "didi" in str(ROOT_PROJECT):
    EXT_PATH = "/media/didi/TOSHIBA EXT/"
else:
    EXT_PATH = "/home/d.lavado/" #cluster data dir


TS40K_PATH = os.path.join(EXT_PATH, 'TS40K-NEW/')

EXPERIMENTS_PATH = os.path.join(ROOT_PROJECT, 'experiments')


WEIGHT_SCHEME_PATH = os.path.join(ROOT_PROJECT, 'core/criterions/hist_estimation.pickle')
#HIST_PATH = os.path.join(SCNET_PIPELINE, "torch_geneo/hist_estimation.pickle")
