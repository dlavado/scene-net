# %%
import math
from pathlib import Path
import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
import laspy as lp
import gc
import torch.nn.functional as F
import torch

import sys

from tqdm import tqdm
import yaml

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

from utils import pcd_processing as eda
from SemKITTI_API.auxiliary.laserscan import SemLaserScan

from VoxGENEO import Voxelization as Vox
from datasets.torch_transforms import ToFullDense, ToTensor, Voxelization

import os

ROOT_PROJECT = Path(os.path.abspath(__file__)).parents[3].resolve()



def build_pole_samples(dataset_path, save_path):
    """
    Builds a new dataset based on the SemanticKITTI dataset with smaller cut out samples in order to increase
    voxel resolution during training.

    Parameters
    ----------

    `dataset_path` - str:
        Path to the semanticKITTI dataset directory

    `save_path` - str:
        Path to the directory of where to save the new samples.

     
    """

    samples_path = os.path.join(save_path, 'samples')

    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    kitti = semKITTI(dataset_path, transform=None)
    counter = 0
    pole_label = 80

    for i in tqdm(range(len(kitti)), desc='Building new dataset...'):

        xyz, gt = kitti[i]
        
        xyz, gt = np.squeeze(xyz), np.squeeze(gt) # get rid of batch dim

        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)
        idx = np.argmax(xyz_max - xyz_min)
        n_steps = 10
        step = int((xyz_max[idx] - xyz_min[idx]) / n_steps)

        for x in np.linspace(xyz_min[idx], xyz_max[idx], step):

            a = np.append(xyz, gt.reshape(-1, 1), axis=1)
            rad = a[np.logical_and(a[:, idx] >= x, a[:, idx] <= x + step)]
            # eda.visualize_ply([eda.np_to_ply(rad[:, :-1])])

            if np.sum(np.isin(rad[:, -1], [80])) >= 5: # if gt contains poles
                # print(rad.shape)
            
                npy_name = os.path.join(samples_path, f'sample_{counter}.npy')

                with open(npy_name, 'wb') as f:
                    np.save(f, rad)  # sample, (x, y, z, label)
                    counter += 1


def crop_tower_samples(xyz:np.ndarray, classes:np.ndarray, obj_class=[80]) -> List[np.ndarray]:

    pcd_tower, _ = eda.select_object(xyz, classes, obj_class)
    towers = eda.extract_towers(pcd_tower, visual=False, eps=5, min_points=10)

    samples = []

    for tower in towers:
        crop, crop_classes = eda.crop_tower_radius(xyz, classes, tower, radius=5)
        tower_section = np.append(crop, crop_classes.reshape(-1, 1), axis=1)
        samples.append(tower_section)

    return samples

def build_pole_radius_samples(dataset_path, save_path):
    """
    Builds a new dataset based on the SemanticKITTI dataset with smaller cut out samples in order to increase
    voxel resolution during training.

    Parameters
    ----------

    `dataset_path` - str:
        Path to the semanticKITTI dataset directory

    `save_path` - str:
        Path to the directory of where to save the new samples.

     
    """

    samples_path = os.path.join(save_path, 'samples')

    if not os.path.exists(samples_path):
        os.makedirs(samples_path)
        counter = 0
    else:
        ans = input('Save Directory already exist. Continue? (y/n)')
        if ans != 'y':
            return
        counter = len(os.listdir(samples_path))

    kitti = semKITTI(dataset_path, transform=None)
    pole_label = 80

    for i in tqdm(range(len(kitti)), desc='Building new dataset...'):

        xyz, gt = kitti[i]
        
        xyz, gt = np.squeeze(xyz), np.squeeze(gt) # get rid of batch dim

        if np.any(gt == pole_label): # if gt contains poles
            pole_samples = crop_tower_samples(xyz, gt, [pole_label])
        else:
            continue

        for sample in pole_samples:
            # print(sample.shape)
            # ply = eda.np_to_ply(sample[:, :-1])
            # eda.color_pointcloud(ply, sample[:, -1])
            # eda.visualize_ply([ply])

            if np.sum(np.isin(sample[:, -1], [pole_label])) >= 5: # if gt contains poles
                npy_name = os.path.join(samples_path, f'sample_{counter}.npy')

                with open(npy_name, 'wb') as f:
                    np.save(f, sample) # sample: (x, y, z, label)
                    counter += 1


class ToTensor:

    def __call__(self, sample):
        pts, labels = sample
        return torch.from_numpy(pts.astype(np.float)), \
               torch.from_numpy(labels.astype(np.float))
    


class semKITTIv2(Dataset):

    def __init__(self, dataset_path, split='samples', transform=ToTensor()):
        """
        Initializes the SemKITTI Dataset for semK pipeline dataset for processed dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the voxelized point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [samples, train, val, test] 
        
        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds

        """

        self.dataset_path = os.path.join(dataset_path, 'samples')
        self.transform = transform
        self.split = split
        self.data_split = {
            'samples' : [0.0, 1.0],   # 100%
            'train' :   [0.0, 0.1],   # 20%
            'val' :     [0.1, 0.3],   # 20%
            'test' :    [0.3, 1.0]    # 60%
        }

        self.KITTI_config_path = os.path.join(ROOT_PROJECT, 'SemKITTI_API', "config/semantic-kitti.yaml")

         # open config file
        try:
            print("Opening config file %s" % self.KITTI_config_path)
            CFG = yaml.safe_load(open(self.KITTI_config_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()


        self.npy_files:np.ndarray = np.array([file for file in os.listdir(self.dataset_path)
                        if os.path.isfile(os.path.join(self.dataset_path, file)) and '.npy' in file])
        
        seed = np.random.randint(0, 123, size=(1,))[0]
        np.random.seed(seed)
        np.random.shuffle(self.npy_files)

        beg, end = self.data_split[split]
        self.npy_files = self.npy_files[math.floor(beg*self.npy_files.size): math.floor(end*self.npy_files.size)]

    def __len__(self):
        return len(self.npy_files)

    def __str__(self) -> str:
        return f"SemKITTIv2 {self.split} Dataset with {len(self)} samples."


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.npy_files[idx])

        if self.split == 'val':
            print(npy_path)

        try:
            npy = np.load(npy_path)
            sample = (npy[:, 0:-1], npy[:, -1]) # xyz-coord (N, 3); label (N,) 

            if self.transform:
                sample = self.transform(sample)
            else:
                sample = (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 
        
        except:
            print(f"Corrupted or Empty Sample")

            if self.transform:
                sample = (np.zeros((100, 3)), np.zeros(100))
                sample = self.transform(sample)
            else:
                sample =  (np.zeros((1, 100, 3)), np.zeros((1, 100))) # batch dim

        return sample
    
    def get_item_no_transform(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.npy_files[idx])

        if self.split == 'val':
            print(npy_path)

        try:
            npy = np.load(npy_path)

            return (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 
        except:
            print(f"Corrupted or Empty Sample")

            return np.zeros((1, 100, 3)), np.zeros((1, 100))

    def get_item_from_path(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, f"sample_{idx}.npy")

        npy = np.load(npy_path)

        return (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 





class semKITTI(Dataset):

    def __init__(self, dataset_path, split='samples', transform=ToTensor()):
        """
        Initializes the SemKITTI Dataset for semK pipeline dataset for original dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the voxelized point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [samples, train, val, test] 
        
        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds

        """
        self.transform = transform
        self.dataset_path = dataset_path


        self.KITTI_config_path = os.path.join(ROOT_PROJECT, 'SemKITTI_API', "config/semantic-kitti.yaml")

         # open config file
        try:
            print("Opening config file %s" % self.KITTI_config_path)
            CFG = yaml.safe_load(open(self.KITTI_config_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()


        self.scan_names = []
        self.label_names = []

        for seq in np.arange(0, 21):

            seq = '{0:02d}'.format(seq)
        
            # does sequence folder exist?
            scan_paths = os.path.join(self.dataset_path, "sequences",
                                        seq, "velodyne")

            print(scan_paths)
            if os.path.isdir(scan_paths):
                print("Sequence folder exists! Using sequence from %s" % scan_paths)
            else:
                print("Sequence folder doesn't exist!")
                continue

            # populate the pointclouds
            self.scan_names += [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_paths)) for f in fn]
            

            label_paths = os.path.join(self.dataset_path, "sequences",
                                        seq, "labels")
            if os.path.isdir(label_paths):
                print("Labels folder exists! Using labels from %s" % label_paths)
            else:
                print("Labels folder doesn't exist!")
                continue

            # populate the pointclouds
            self.label_names += [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_paths)) for f in fn]

        self.scan_names = np.sort(np.array(self.scan_names))
        self.label_names = np.sort(np.array(self.label_names))

        assert len(self.scan_names) == len(self.label_names)

        color_dict = CFG["color_map"]
        self.semKITTIapi = SemLaserScan(len(color_dict), color_dict, project=False)

        data_split = {
            'samples' : [0.0, 1.0],   # 100%
            'train' :   [0.0, 0.2],   # 20%
            'val' :     [0.2, 0.4],   # 20%
            'test' :    [0.4, 1.0]    # 60%
        }

        beg, end = data_split[split]
        self.scan_names = self.scan_names[math.floor(beg*self.scan_names.size): math.floor(end*self.scan_names.size)]
        self.label_names = self.label_names[math.floor(beg*self.label_names.size): math.floor(end*self.label_names.size)]


    def __len__(self):
        return len(self.scan_names)

   
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #loading scan into semKITTI api
        self.semKITTIapi.open_scan(self.scan_names[idx])
        self.semKITTIapi.open_label(self.label_names[idx])

        xyz = self.semKITTIapi.points
        gt = self.semKITTIapi.sem_label

        assert len(xyz) == len(gt)

        sample = (xyz, gt) # xyz-coord (N, 3); label (N,) 

        try:
            if self.transform:
                sample = self.transform(sample)
            else:
                sample = (xyz[None], gt[None, :]) # xyz-coord (1, N, 3); label (1, N) 
        
        except:
            print(f"Corrupted or Empty Sample")

            if self.transform:
                sample = (np.zeros((100, 3)), np.zeros(100))
                sample = self.transform(sample)
            else:
                sample =  (np.zeros((1, 100, 3)), np.zeros((1, 100))) # batch dim

        return sample
    




# %%
def main():
    
    EXT_DIR = "/media/didi/TOSHIBA EXT/"
    SEMK_DATA_PATH = os.path.join(EXT_DIR, 'SemKITTI/SemKITTI/dataset')

    NEW_SEMK_PATH = os.path.join(EXT_DIR, 'SemKITTI')

    input("build?")
    build_pole_radius_samples(SEMK_DATA_PATH, NEW_SEMK_PATH)


    KITTI_config_path = os.path.join(ROOT_PROJECT, 'SemKITTI_API', "config/semantic-kitti.yaml")

    # open config file
    try:
        print("Opening config file %s" % KITTI_config_path)
        CFG = yaml.safe_load(open(KITTI_config_path, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    SemKITTI_labels = CFG['labels']
    pole_label = list(SemKITTI_labels.keys())[list(SemKITTI_labels.values()).index('pole')]
    #traffic_sign_label = list(SemKITTI_labels.keys())[list(SemKITTI_labels.values()).index('traffic-sign')]
    
    vxg_size  = (64, 64, 256)
    vox_size = (0.5, 0.5, 0.2) #only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([pole_label], vxg_size=vxg_size, vox_size=vox_size), 
                        ToTensor(), 
                        ToFullDense()])
    
    semK = semKITTIv2(dataset_path=NEW_SEMK_PATH, split='train', transform=composed)

    print(len(semK))

    for i in range(5, len(semK)):
        print(i)
        vox, vox_gt = semK[i]
        print(vox.shape, vox_gt.shape)

        # Vox.plot_voxelgrid(torch.squeeze(vox))
        # Vox.plot_voxelgrid(torch.squeeze(vox_gt))
        
        Vox.visualize_pred_vs_gt(torch.squeeze(vox), torch.squeeze(vox_gt), plot=False)
        input('Continue?')
        

if __name__ == "__main__":
    main()

# %%
