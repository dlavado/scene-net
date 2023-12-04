import math
from pprint import pprint
from tracemalloc import start
from typing import Any, List, Mapping, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection, JaccardIndex, Precision, Recall, F1Score, FBetaScore, AUROC, AveragePrecision
from torch.utils.data import DataLoader
from torchviz import make_dot
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image


import sys
import time
from pathlib import Path

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from utils import pcd_processing as eda
from utils import Voxelization as Vox
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##################
# Plotting Utils #
##################


def rem_axis(ax, sides = ['top', 'right', 'left', 'bottom']):
    av_sides = ['top', 'right', 'left', 'bottom']
    # Remove axes splines
    for s in sides:
        ax.spines[s].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    diff  = list(set(av_sides) - set(sides))
    if diff == []:
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
    else:
        for side in diff:
            if side in ['left', 'right']:
                ax.yaxis.set_ticks_position(side)
            else:
                ax.xaxis.set_ticks_position(side)


def plot_geneo_params(csv_file, save_dir=None):
    df = pd.read_csv(csv_file)

    params = df['name'].unique()
    x_data = df['epoch'].unique()

    g_unq, g_count = np.unique(np.array([p.split('.')[1] for p in params if not 'lambda' in p]), return_counts=True)
    cvxs = [g for g in params if 'lambda' in g]

    ############################## GENEO Coeffs ####################################

    plt.clf()
    fig = plt.figure()
    ax = plt.subplot(111)
    rem_axis(ax, ['top', 'right']) 

    for cvx in cvxs:
        y_data = df[df['name'] == cvx]['value']
        plt.plot(x_data, y_data, label=cvx.split('.')[1])
    plt.title("convex coefficients")
    plt.legend()

    if save_dir is None:
        plt.show() 
    else:
        plt.savefig(os.path.join(save_dir, "cvx_coeffs.png"), dpi=100, facecolor='white', transparent=False)

    plt.close()

    ############################## GENEO Params ####################################

    fig = plt.figure()
    ax = plt.subplot(111)
    rem_axis(ax, ['top', 'right'])  

    rows = len(g_unq)
    figure, axis = plt.subplots(rows, np.max(g_count),figsize=(2*np.max(g_count), 2*rows), dpi=120)
    plt.tight_layout()

    for i, p in enumerate(params):
        y_data = df[df['name'] == p]['value']

        if not 'lambda' in p:
            spl = p.split('.')
            idx = np.where(g_unq == spl[1])[0][0]
            title = spl[1] + "_" + spl[-1]

            axis[idx, i % g_count[idx]].plot(x_data, y_data)
            axis[idx, i % g_count[idx]].set_title(title)
    if save_dir is None:
        plt.show() 
    else:
        plt.savefig(os.path.join(save_dir, "geneo_params.png"), dpi=100, facecolor='white', transparent=False)
    plt.close()


def merge_imgs(f_name:str, images:list):
    """
    Saves a single image file with the images name provided in the images array.
    """
    images = [Image.open(x) for x in images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x = 0
    for im in images:
        new_image.paste(im, (x,0))
        x += im.size[0]
    new_image.save(f'{f_name}.jpg')


def plot_metric(metric:np.ndarray, save_dir:str, title:str, legend=Union[List[str], None]):
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111)
    rem_axis(ax, ['top', 'right'])  

    if len(metric.shape) == 1: # if `metric` is 1D, make it 2D
        metric = metric[..., None]

    for i in range(metric.shape[-1]):
        num_epochs = np.arange(1, len(metric != 0)+1) # num_epochs
        plt.plot(num_epochs, metric[:, i])

        for x, y in zip(num_epochs, metric[:, i]):
            if x in np.linspace(1, len(metric != 0), 5, dtype=np.int):
                label = "{:.3f}".format(y)

                plt.annotate(label, # this is the text
                            (x,y), # these are the coordinates to position the label
                            textcoords="offset points", # how to position the text
                            xytext=(0,10), # distance from text to points (x,y)
                            ha='center') # horizontal alignment can be left, right or center

    plt.legend(legend)
    plt.xlabel('epochs')
    plt.ylabel('metric')
    plt.title(title)
    plt.savefig(os.path.join(save_dir, f"{title}.png"))


##################
# Observer Utils #
##################


def transform_forward(batch, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return batch[0].to(device), batch[1].to(device)

def transform_metrics(pred:torch.Tensor, target:torch.Tensor):
    return torch.flatten(pred), torch.flatten(target).to(torch.int)


def process_batch(gnet, batch, geneo_loss, opt, metrics, requires_grad=True):
    batch = transform_forward(batch)
    loss, pred = forward(gnet, batch, geneo_loss, opt, requires_grad)
    if metrics is not None:
        pred, targets = transform_metrics(pred, batch[1])
        metrics(pred, targets)
    return loss, pred

def forward(gnet:torch.nn.Module, batch, geneo_loss:torch.nn.Module, opt : Union[torch.optim.Optimizer, None], requires_grad=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a forward pass of `gnet` with data `batch`, loss `geneo_loss` and optimizer `opt`.

    if `requires_grad`, then it computes the backwards pass through the network.

    Returns
    -------
    `loss` - float:
        loss value computed with `geneo_loss`

    `pred` - torch.tensor:
        gnet's prediction

    """
    # --- Data to GPU if available ---
    vox, vox_gt = batch
    vox = vox.to(device)
    vox_gt = vox_gt.to(device)

    # --- Forward pass ---
    #start = time.time()
    pred = gnet(vox)
    #end = time.time()
    #print(f"Prediction inference time: {end - start}")

    loss = geneo_loss(pred, vox_gt, gnet.get_cvx_coefficients(), gnet.get_geneo_params()) 

    #print(loss)

    # --- Backward pass ---
    if requires_grad:
        opt.zero_grad()
        loss.backward()
        #print([(n, p.grad.item()) if p.grad is not None else (n, None) for n, p in gnet.named_parameters() ])
        opt.step()

    return loss, pred


def init_metrics(tau=0.65):
    return MetricCollection([
        JaccardIndex(num_classes=2, threshold=tau),
        Precision(threshold=tau),
        Recall(threshold=tau),
        F1Score(threshold=tau),
        FBetaScore(beta=0.5, threshold=tau),
        #AveragePrecision(),
        #Accuracy(threshold=tau),
        #AUROC() # Takes too much GPU memory
        #BinnedAveragePrecision(num_classes=1, thresholds=torch.linspace(0.5, 0.95, 20))
    ]).to(device)


def load_state_dict(model_path, gnet_class, model_tag='loss', kernel_size=None) -> Union[None, Tuple[SCENE_Net, Mapping[str, Any]]]:
    """
    Returns GENEO-Net model and model_checkpoint
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
                              plot=False).to(device)
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



def to_numpy(tensor:torch.Tensor, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Turns a torch Tensor to numpy
    """
    if device != 'cpu':
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def plot_voxelgrid(tensor:torch.Tensor, title="", color_mode='density', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if tensor is None:
        ValueError("Tensor is None")
    if torch.sum(tensor > 0) == 0:
        ValueError("Empty Tensor")
    else:
        Vox.plot_voxelgrid(to_numpy(torch.squeeze(tensor), device), color_mode=color_mode, title=title)


def visualize_batch(vox_input:torch.Tensor=None, gt:torch.Tensor=None, pred:torch.Tensor=None, idx:int=None, tau=0.7,
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    #print(f"Showing random sample in batch...\n")
    batch_size = pred.shape[0]
    
    # selecting random sample from batch
    if idx is None:
        idx = np.random.randint(0, batch_size, size=1)[0]
   
    #plot_voxelgrid(vox_input[idx], title='Input Voxelgrid')

    plot_voxelgrid(pred[idx], color_mode='ranges', title='GENEO-Net Probability Prediction')
    plot_voxelgrid(Vox.prob_to_label(pred[idx], tau), title='GENEO-Net Prediction')


    visualize_regression(pred[idx], gt[idx], tau, device)


def visualize_regression(pred:torch.Tensor=None, gt:torch.Tensor=None, tau=0.7, 
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    if pred is not None:
        pred = torch.squeeze(pred)
        if len(pred[pred > 0]) == 0:
            print(f"Prediction is Empty!")
    else:
        ValueError(f"Prediction is None")
    
    if gt is None:
         ValueError(f"Ground Truth is None")

    gt = torch.squeeze(gt)
    pred = Vox.prob_to_label(pred, tau)
    pred, gt = to_numpy(pred, device), to_numpy(gt, device)

    Vox.visualize_pred_vs_gt(pred.astype(np.int), gt.astype(np.int))


def visualize_quantiles(vox_input:torch.Tensor=None, pred:torch.Tensor=None,
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    plot_voxelgrid(vox_input, title='Input Voxelgrid')
    Vox.plot_quantile_uncertainty(to_numpy(torch.squeeze(pred), device), legend=True)



def scnet_calibration(model_path, gnet_class, ts40k_val, batch_size, mode="TempScaling"):
  
    gnet, _ = load_state_dict(model_path, gnet_class, model_tag='FBetaScore')
    
    # --- Dataset Initialization ---
    ts40k_val_loader = DataLoader(ts40k_val, batch_size=batch_size, shuffle=True, num_workers=4)


    # --- Temperature Scaling ---
    if mode == 'TempScaling':
        from scenenet_pipeline.calibration.temperature_scaling import ModelWithTemperature
        scaled_model = ModelWithTemperature(gnet)
        print(f"\n\nInitializing Temperature Scaling...")
        scaled_model.set_temperature(ts40k_val_loader)
    elif mode == 'LogRegression':
        from scenenet_pipeline.calibration import log_regression
        import functools
        #in_features = functools.reduce(lambda x,y: x*y, (64, 64, 64)) #too much memory
        in_features = 1000
        log_regression.calibrate(gnet, in_features, ts40k_val_loader, epochs=50)
    else:
        ValueError(f"Mode {mode} is not Implemented; Available Modes = [TempScaling, LogRegression]")





class EarlyStopping():
    def __init__(self, tolerance=10, metric='loss', min_delta=0):

        self.tolerance = tolerance
        self.best_metric = 1e10 if metric == 'loss' else 0
        self.min_delta = min_delta
        self.metric = metric
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_metric):
        if self.metric == 'loss':
            if train_metric >= self.best_metric:
                self.counter +=1
                if self.counter >= self.tolerance:  
                    self.early_stop = True
            else:
                self.best_metric = train_metric
                self.counter = 0
        else:
            if train_metric <= self.best_metric:
                self.counter +=1
                if self.counter >= self.tolerance:  
                    self.early_stop = True
            else:
                self.best_metric = train_metric
                self.counter = 0



#######################
# Compute Tower Coord #
#######################

def extract_towers(vxg:torch.Tensor, eps=10, min_points=50, print=False) -> Union[List[np.ndarray], np.ndarray]:
    xyz = Vox.vxg_to_xyz(vxg)
    xyz_pcd = eda.np_to_ply(xyz)
    if print:
        eda.visualize_ply([xyz_pcd], window_name="Prediction in Raw 3D Point Cloud Format")
    if len(xyz) > 0:
        towers = eda.extract_towers(xyz_pcd, eps=eps, min_points=min_points, visual=print) #performs DBSCAN
        if len(towers) == 0: # no extracted towers
            return [], []
    else:
        return [], []

    return towers, np.vstack([eda.xyz_centroid(t) for t in towers])



def compute_euc_dists(vxg:torch.Tensor, gt:torch.Tensor, min_dist=3.5, min_points=18, visual=False):
    """
    Extracts the tower candidates from the prediction `vxg` and the ground-truths from `gt`
    by the means of DBSCAN. \\
    Then, for each gt_tower, we find the closest tower_candidate and compute their Euclidean
    Distance (the z-coord is disregarded)

    Parameters
    ----------
    `vxg` - torch.Tensor:
        (4, voxelgrid_dims) torch tensor with coords + predicted label
    `gt` - torch.Tensor:
        (4, voxelgrid_dims) torch tensor with coord + ground_truth
    
    Returns
    -------
    `sample_distances` - list with [gt_centroid, closest_tower__candidate_centroid, their_euclidean_distance]\\
    if there are no tower candidates, then closest_tower__candidate_centroid is `None`
    """

    _, vxg_centroids = extract_towers(vxg, eps = min_dist, min_points=min_points,print=visual)
    _, gt_centroids = extract_towers(gt, eps = min_dist, min_points=min_points,print=visual)


    if len(vxg_centroids) > 0: # There are towers in the prediction `vxg`
        vxg_centroids = vxg_centroids[:, :-1] # disregard z coord
        gt_centroids = gt_centroids[:, :-1] # disregard z coord

        filtered_centroids = None
        for vxg_c in vxg_centroids: # centroids that are at an euclidean distance of less than 1.5 are considered part of the same tower
            c_full = np.full_like(vxg_centroids, vxg_c)
            euc_dists = np.linalg.norm(c_full - vxg_centroids, axis=1)

            new_point = np.mean(vxg_centroids[euc_dists <= 1.5], axis=0)

            if filtered_centroids is None:
                filtered_centroids = [new_point]
            else:
                filtered_centroids = np.concatenate((filtered_centroids, [new_point]))

        filtered_centroids = np.unique(filtered_centroids, axis=0)
        vxg_centroids = filtered_centroids

        sample_distances = [] # holds for each gt_tower, (gt_c, vxg_c, min_euc_dist)
        for gt_c in gt_centroids: # finding the closest vxg_tower in for each gt_tower
            gt_full = np.full_like(vxg_centroids, gt_c)
            euc_dists = np.linalg.norm(gt_full - vxg_centroids, axis=1) 
            argmin = np.argmin(euc_dists)
            sample_distances.append((gt_c, vxg_centroids[argmin], euc_dists[argmin]))   
    else:
        sample_distances = [(gt_c, None, 0) for gt_c in gt_centroids]

    if visual:
        if len(vxg_centroids) != len(gt_centroids):
            print(f"False Positives in Coordinate Predictions:")
            print(f"Predictions:\t\t\tGround Truth:\n{vxg_centroids}\t\t\t{gt_centroids}")
        
        print(f"Number of tower coordinate predictions: {len(vxg_centroids)}")
        print(f"Number of towers in the Ground Truth: {len(gt_centroids)}")

    return sample_distances


def aggregate_centroids(vxg_centroids):

    filtered_centroids = None

    min_euc = 1.5

    if len(vxg_centroids) == 0:
        return np.empty((0, 2)) # empty array to maintain the format of the pipeline

    vxg_centroids = vxg_centroids[:, :-1]

    for i, vxg_c in enumerate(vxg_centroids): # centroids that are at an euclidean distance of less than 1.5 are considered part of the same tower
        c_full = np.full_like(vxg_centroids, vxg_c)
        euc_dists = np.linalg.norm(c_full - vxg_centroids, axis=1)

        new_point = np.mean(vxg_centroids[euc_dists <= min_euc], axis=0)

        if filtered_centroids is None:
            filtered_centroids = [new_point]
        else:
            filtered_centroids = np.concatenate((filtered_centroids, [new_point]))

    filtered_centroids = np.unique(filtered_centroids, axis=0)

    return filtered_centroids


def filter_towers(vxg:torch.Tensor, towers:list, centroids:np.ndarray, threshold:float, visual=True):
    """
    Disregards any tower cluster that has a xy variance above a certain threshold.
    The purpose of this operation is to disregard wall clusters deemed as towers

    Parameters
    ----------

    `towers` - list[np.ndarray]:
        list with tower proposals with size C;

    `centroids` - np.ndarray:
        numpy array in (C, 3) format containing C centroids of towers

    `threshold` - float:
        xy variance threshold that distinguishes between walls and towers
    """
   
    keep = np.zeros(len(towers), dtype=np.bool)
    tower_height = 14 # avg tower height established during EDA phase
    radius = 15 # radius used to cut Labelec samples

    xyz_center = np.mean(Vox.vxg_to_xyz(vxg), axis=0)

    if visual:
        print(f"Voxel-grid center: {xyz_center}")

    for i, t in enumerate(towers):
        t_min = np.min(t, axis=0)
        t_max = np.max(t, axis=0)
        xy_var = np.max(t_max[:-1] - t_min[:-1])

        t_height = t_max[-1] - t_min[-1]
        if t_height >= tower_height: # centroids with enough height are considered towers 
            keep[i] = True
        else:
            # xy_var = np.sum(t_max - t_min)
            keep[i] = xy_var <= threshold #centroids with a big enough xy variance are deemed as walls

        # Any centroid that is at the border of the point cloud is not a tower
        keep[i] = keep[i] and np.sum((centroids[i][:-1] - xyz_center[:-1])**2) <= (radius - threshold*2)**2

        #print((centroids[i][:-1] - xyz_center[:-1])**2, (radius - threshold*2)**2)
        if visual:
            print(f"centroid {centroids[i]} xy_var: {xy_var:.4f}; kept? {keep[i]}")

    return [towers[i] for i in range(len(towers)) if keep[i]], centroids[keep]
        

   



def get_tower_proposals(xyz:torch.Tensor, dens:torch.Tensor, pred:torch.Tensor,
                       min_dist=3.5, min_points=18, visual=False):
    """
    Same behavior as `compute_euc_dists` but without the performance measure
    """

    pred_vxg = torch.cat((xyz, pred))

    towers, vxg_centroids = extract_towers(pred_vxg, eps = min_dist, min_points=min_points,print=visual)


    for i in range(len(towers)): # remove buggy centroid
        if np.all(vxg_centroids[i] == [0., 0., 0.]):
            vxg_centroids = np.delete(vxg_centroids, i, axis=0)
            towers.pop(i)
            print(f"Faulty centroid removed...")
            break

    if len(towers) >= 1: # if there are tower proposals
        towers, vxg_centroids = filter_towers(torch.cat((xyz, dens)), towers, vxg_centroids, min_dist/2) # to get rid of wall proposals

    vxg_centroids = aggregate_centroids(vxg_centroids)

    if visual:
        print(f"filtered vxg_centroids:\n {vxg_centroids}")

    return vxg_centroids


def plot_centroids(vxg:torch.Tensor, dens:torch.Tensor, centroids:np.ndarray):
    """

    plots centroids proposals in a point cloud

    Parameters
    ----------

    `vxg` - torch.Tensor:
        Input voxel-grid to the model

    `centroids` - np.ndarray:
        (N, 2) matrix with N xy centroids to plot
    """

    dens = Vox.prob_to_label(dens, 0.01)

    xyz = Vox.vxg_to_xyz(torch.cat((vxg, dens)))

    if len(xyz) == 0:
        return

    classes = np.zeros(len(xyz))

    z_min, z_max = np.min(xyz, axis=0)[-1], np.max(xyz, axis=0)[-1]
    # print(z_min, z_max)
    z_points = 100
    lin = np.linspace(z_min, z_max, num=z_points)

    for i in range(len(centroids)):
        c_full = np.full((z_points, 2), fill_value=centroids[i])
        c_full = np.concatenate((c_full, lin.reshape(-1, 1)), axis=1) # add z_coord
        # print(c_full)
        # c_full = np.concatenate((c_full, np.full_like(lin.reshape(-1, 1), fill_value=4)), axis=1) # add class
        # print(c_full.shape)
        xyz = np.concatenate((xyz, c_full), axis=0)
        # classes = np.concatenate((classes, np.full_like(lin, fill_value=eda.POWER_LINE_SUPPORT_TOWER)))
        classes = np.concatenate((classes, np.full_like(lin, fill_value=i+1)))

    
    #print(xyz.shape)

    xyz_pcd = eda.np_to_ply(xyz)

    cluster_colors = eda.color_pointcloud(xyz_pcd, classes)
    cluster_colors = np.hstack((cluster_colors, classes[:, None])) # (N, 4) -> r, g, b, cluster_idx
    cluster_colors = np.unique(cluster_colors, axis=0)    

    print(f"\nPlotting Centroids...\nColor Scheme:")
    for color_idx in cluster_colors:
        color = color_idx[:-1]
        idx = np.int(color_idx[-1])
        _, color_name = Vox.get_colour_name(color)

        if idx == 0: # original pcd color idx == 0
            print(f"\t3D Point Cloud Color: {color_name}")
        else:
            # print(idx, cluster_colors[idx])
            print(f"\tCentroid {centroids[idx-1]} plotted with color: {color_name}")

    

    eda.visualize_ply([xyz_pcd], window_name="Prediction Point Cloud")
    







        

def find_best_gnet(models_root, model_tag=None):
    """
    Searches along `models_root` to find the best model according to different testing metrics.

    It looks up the testing results of each GENEO_Net model and print the best found metrics and
    the model's path.

    Parameters
    ----------
    `models_root` - str:
        path of the directory that contains all trained and tested models. Leaf directories must contain
        a `gnet.pt` file with the respective model.
    
    `model_tag` - str:
        states which tag we are interested in; The methods will return the model_path of the gnet that maximizes
        such tag.

    Returns
    -------
        The best GENEO-Net model path

    """

    best_res = {
        'loss': 1e10,
        'Precision': 0,
        'Recall': 0,
        'F1Score': 0, 
        'FBetaScore': 0,
        'JaccardIndex': 0
    }

    best_models_path = best_res.copy()

    for root, _, files in os.walk(models_root):

        if 'gnet.pt' in files: # root contains a gnet model
            model_path = os.path.join(root, 'gnet.pt')
            state_dict = torch.load(model_path)
            if 'model_props' not in state_dict.keys() or 'test_results' not in state_dict['model_props'].keys():
                continue

            test_results = state_dict['model_props']['test_results']

            for tag in best_res:
                test_res = test_results.get(tag, None)
                if isinstance(test_res, dict):
                    performance = float(test_res[tag])
                else:
                    performance = 0 if test_res is None else float(test_results[tag])

                if tag == 'loss':
                    if performance <= best_res[tag]:
                        best_res[tag] = performance
                        best_models_path[tag] = root
                elif tag in best_res: # else they're metrics and we want to maximize them
                    if performance >= best_res[tag]: 
                        best_res[tag] = performance
                        best_models_path[tag] = root

    if model_tag is None:
        pprint(best_res)
        print()
        pprint(best_models_path)
    else:
        return best_models_path[model_tag]


