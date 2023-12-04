# %%
import math
from pathlib import Path
import random
import shutil
from time import sleep
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
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from utils import pcd_processing as eda
from core.datasets.torch_transforms import ToFullDense, Voxelization, ToTensor

from utils import voxelization as Vox
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_data_samples(data_dirs:List[str], save_dir=os.getcwd(), tower_radius=True, data_split:dict={"fit": 0.6, "test": 0.4}):
    """

    Builds a dataset of voxelized point clouds in npy format according to 
    the las files in data_dirs and saves it in save_dir.

    Parameters
    ----------

    `data_dirs` - str list : 
        list of directories with las files to be converted

    `save_dir` - str:
        directory where to save the npy files.

    `tower_radius` - bool:
        type of sample we want to build -> tower_radius or two_towers

    `data_split` - dict {"str": float} or int:
        if data_split is dict:
            split of the dataset into sub-folders; keys correspond to the folders and the values to the sample split
        elif data_split is int:
            data_split == 0 for no dataset split 
    """
    
    for folder in data_split.keys():
        if not os.path.exists(os.path.join(save_dir, folder)):
            os.makedirs(os.path.join(save_dir, folder))

    # process all files to `fit_path` and then retreive test samples
    fit_path = os.path.join(save_dir, 'fit')
    counter = len(os.listdir(fit_path)) 

    read_files = []
    pik_name = 'read_files.pickle'
    pik_path = os.path.join(save_dir, pik_name)
    if os.path.exists(pik_path):
        read_files = eda.load_pickle(pik_path)

    for dir in data_dirs:
        os.chdir(dir)

        for las_file in os.listdir("."):
            filename = os.getcwd() + "/" + las_file

            if ".las" in filename:
                las = lp.read(filename)
            else:
                continue

            if filename in read_files:
                continue

            print(f"\n\n\nReading...{filename}")

            xyz, classes = eda.las_to_numpy(las)

            if np.any(classes == eda.POWER_LINE_SUPPORT_TOWER):
                if tower_radius:
                    t_samples = eda.crop_tower_samples(xyz, classes)
                else:
                    t_samples = eda.crop_two_towers_samples(xyz, classes)
            else:
                continue

            #f_samples = eda.crop_ground_samples(xyz, classes)
            f_samples = []
            file_samples = f_samples + t_samples
            print(f"file samples: {len(file_samples)} (tower, no_tower): ({len(t_samples)}, {len(f_samples)})")

            for sample in file_samples:
                try:             
                    npy_name = f'{fit_path}/sample_{counter}.npy'
                    print(npy_name)

                    with open(npy_name, 'wb') as f:
                        np.save(f, sample)  # sample: x, y, z, class
                        counter += 1
                except:
                    print(f"Problem occurred while reading {filename}\n\n")
                    continue
                
                gc.collect()

            del las
            del t_samples
            gc.collect()
            read_files.append(filename)
            eda.save_pickle(read_files, pik_path)
          

    if data_split == 0:
        return # all files in fit_path

    samples = os.listdir(fit_path)
    random.shuffle(samples)

    assert sum(list(data_split.values())) <= 1, "data splits should not surpass 1"

    split_sum = 0
    sample_size = len(samples)
    print(f"Number of total samples: {sample_size}")

    for folder, split in data_split.items():
        if folder == 'fit':
            split_sum += split
            continue

        dir = os.path.join(save_dir, folder)
        if not os.path.exists(dir):
            os.makedirs(dir)

        samples_split = samples[int(split_sum*sample_size):math.ceil((split_sum+split)*sample_size)]
        split_sum += split
        print(f"Samples in {folder}: {len(samples_split)}")

        for sample in samples_split:
            shutil.move(os.path.join(fit_path, sample), dir)





class TS40K(Dataset):

    def __init__(self, dataset_path, split='fit', transform=None) -> None:
        """
        Initializes the TS40K dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the voxelized point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [fit, test] 
            
        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds
        """
        super().__init__()

        self.transform = transform
        self.split = split

        self.dataset_path = os.path.join(dataset_path, split)

        self.npy_files:np.ndarray = np.array([file for file in os.listdir(self.dataset_path)
                        if os.path.isfile(os.path.join(self.dataset_path, file)) and '.npy' in file])
        

    def __len__(self):
        return len(self.npy_files)

    def __str__(self) -> str:
        return f"TS40K {self.split} Dataset with {len(self)} samples"

    def set_transform(self, new_transform):
        self.transform = new_transform
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.npy_files[idx])

        try:
            npy = np.load(npy_path)
        except:
            print(f"Unreadable file: {npy_path}, loading random sample instead...")
            npy_path = os.path.join(self.dataset_path, self.npy_files[random.randint(0, len(self))])
            npy = np.load(npy_path)

        sample = (npy[:, 0:-1], npy[:, -1]) # xyz-coord (N, 3); label (N,) 

        while True: # there is a bug with the dataloader or the data is corrupted, idk why it happens

            try:
                if self.transform:
                    return self.transform(sample)
                else:
                    return (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 
            
            except:

                print(f"Corrupted or Empty Sample: {npy_path}, loading random sample instead...")

                npy_path = os.path.join(self.dataset_path, self.npy_files[random.randint(0, len(self))])
                npy = np.load(npy_path)
                sample = (npy[:, 0:-1], npy[:, -1])

        return sample
            
            

def main():
    
    ROOT_PROJECT = Path(os.path.abspath(__file__)).parents[3].resolve()

    #EXT_DIR = "/media/didi/TOSHIBA EXT/LIDAR/"
    EXT_DIR = "/media/didi/TOSHIBA EXT"
    SAVE_DIR = "/media/didi/TOSHIBA EXT/TS40K-NEW/"

    build_data_samples([os.path.join(EXT_DIR, 'LIDAR'), os.path.join(EXT_DIR, 'Labelec_LAS')], SAVE_DIR)

    #composed = Compose([ToTensor(), AddPad((3, 3, 3, 3, 3, 3))]) 
    vxg_size  = (64, 64, 64)
    vox_size = (0.5, 0.5, 0.5) #only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size), 
                        ToTensor(), 
                        ToFullDense(apply=(True, False))])
    
    ts40k = TS40K(dataset_path=EXT_DIR, split='fit', transform=composed)

    print(len(ts40k))

    print(ts40k[0])
    vox, vox_gt = ts40k[0]
    print(vox.shape, vox_gt.shape)

    Vox.plot_voxelgrid(torch.squeeze(vox))
    Vox.plot_voxelgrid(torch.squeeze(vox_gt))

    # Hyper parameters
    NUM_EPOCHS = 5
    NUM_SAMPLES = len(ts40k)
    BATCH_SIZE = 1
    NUM_ITER = math.ceil(NUM_SAMPLES / BATCH_SIZE)

    ts40k_loader = DataLoader(ts40k, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for epoch in range(NUM_EPOCHS):
        for i, (xyz, vox, vox_gt) in tqdm(enumerate(ts40k_loader), desc=f"epoch {epoch+1}", ): 
            #print(f"epoch {epoch} / {NUM_EPOCHS}; step {i+1} / {NUM_ITER}; input: {vox.shape}; labels: {vox_gt.shape}")
            print(xyz.shape)
            print(vox.shape)
            print(vox_gt.shape)

            xyz, vox, vox_gt = xyz[0][0], vox[0][0], vox_gt[0][0]  #1st sample of batch; decapsulate

            Vox.plot_voxelgrid(vox)
            Vox.plot_voxelgrid(vox_gt)
            xyz_gt = Vox.vxg_to_xyz(torch.cat((xyz, vox_gt[None]), axis=0))
            pcd_gt = eda.np_to_ply(xyz_gt)
            eda.visualize_ply([pcd_gt])

            print(f"xyz centroid = {eda.xyz_centroid(xyz_gt)}")
            print(f"euc dist = {eda.euclidean_distance(xyz_gt, xyz_gt)}")


if __name__ == "__main__":
    main()

# %%
