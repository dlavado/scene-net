
from typing import Tuple
import torch
import numpy as np
import torch.nn.functional as F
from utils import voxelization as Vox
import utils.pcd_processing as eda

class ToTensor:

    def __call__(self, sample):
        sample = list(sample)
        return tuple([torch.from_numpy(s.astype(np.float)) for s in sample])



class ToFullDense:
    """
    Transforms a Regression Dataset into a Belief Dataset.

    Essentially, any voxel that has tower points is given a belief of 1,
    in order to maximize the towers' geometry.
    For the input, the density is normalized to 1, so empty voxels have a value
    of 0 and 1 otherwise.

    It requires a discretization of raw LiDAR Point Clouds in Torch format.
    """

    def __init__(self, apply=[True, True]) -> None:
        
        self.apply = apply
    
    def densify(self, tensor:torch.Tensor):
        return (tensor > 0).to(tensor)

    def __call__(self, sample:torch.Tensor):

        vox, gt = [self.densify(tensor) if self.apply[i] else tensor for i, tensor in enumerate(sample) ]
        
        return vox, gt



class Voxelization:

    def __init__(self, keep_labels, vox_size:Tuple[int]=None, vxg_size:Tuple[int]=None) -> None:
        """
        Voxelizes raw LiDAR 3D point points in `numpy` (N, 3) format 
        according to the provided discretization

        Parameters
        ----------
        `vox_size` - Tuple of 3 Ints:
            Size of the voxels to dicretize the point clouds
        `vxg_size` - Tuple of 3 Ints:
            Size of the voxelgrid used to discretize the point clouds

        One of the two parameters need to be provided, `vox_size` takes priority

        Returns
        -------
        A Voxelized 3D point cloud in Density/Probability mode
        """
        
        if vox_size is None and vxg_size is None:
            ValueError("Voxel size or Voxelgrid size must be provided")


        self.vox_size = vox_size
        self.vxg_size = vxg_size
        self.keep_labels = keep_labels


    def __call__(self, sample:np.ndarray):
        
        pts, labels = sample

        voxeled_xyz = Vox.hist_on_voxel(pts, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)
        voxeled_gt = Vox.reg_on_voxel(pts, labels, self.keep_labels, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)

        return voxeled_xyz[None], voxeled_gt[None] # vox-point-density, vox-tower-prob



class AddPad:

    def __init__(self, pad:Tuple[int]):
        """
        `pad` is a tuple of ints that contains the pad sizes for each dimension in each direction.\n
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4]) 
        """
        self.p3d = pad

    def __call__(self, sample):
        pts, labels = sample
        return F.pad(pts, self.p3d, 'constant', 0), F.pad(labels, self.p3d, 'constant', 0)


    


# ---- Centroid Versions


class xyz_ToFullDense:
    """
    Transforms a Regression Dataset into a Belief Dataset.

    Essentially, any voxel that has tower points is given a belief of 1,
    in order to maximze the towers' geometry.
    For the input, the density is notmalized to 1, so empty voxels have a value
    of 0 and 1 otherwise.

    It requires a discretization of raw LiDAR Point Clouds in Torch format.
    """

    def __call__(self, sample:torch.Tensor):
        xyz, dense, labels = sample

        return xyz, (dense > 0).to(dense), (labels > 0).to(labels) #full dense
        

class xyz_Voxelization:

    def __init__(self, keep_labels, vox_size:Tuple[int]=None, vxg_size:Tuple[int]=None) -> None:
        """
        Voxelizes raw LiDAR 3D point points in `numpy` (N, 3) format 
        according to the provided discretization

        Parameters
        ----------
        `vox_size` - Tuple of 3 Ints:
            Size of the voxels to dicretize the point clouds
        `vxg_size` - Tuple of 3 Ints:
            Size of the voxelgrid used to discretize the point clouds

        One of the two parameters need to be provided, `vox_size` takes priority

        Returns
        -------
        A Voxelized 3D point cloud in Density/Probability mode
        """
        
        if vox_size is None and vxg_size is None:
            ValueError("Voxel size or Voxelgrid size must be provided")


        self.vox_size = vox_size
        self.vxg_size = vxg_size
        self.label = keep_labels


    def __call__(self, sample:np.ndarray):
        
        pts, labels = sample

        voxeled_xyz = Vox.centroid_hist_on_voxel(pts, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)
        voxeled_gt = Vox.centroid_reg_on_voxel(pts, labels, self.label, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)

        assert np.array_equal(voxeled_xyz[:-1], voxeled_gt[:-1])

        return voxeled_xyz[None, :-1], voxeled_xyz[None, -1], voxeled_gt[None, -1] # xyz-vox-centroid, vox-point-density, vox-tower-prob
