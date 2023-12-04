"""
In this file we will import the LiDAR data
into dataframes, validate and transform the data
in order to perform EDA on it.

@author: d.lavado

"""
# %%
from locale import normalize
from pprint import pprint
import random
from typing import List, Union
import numpy as np
import pandas as pd
import os
import laspy as lp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import open3d as o3d
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import cloudpickle
from math import ceil, degrees

from sympy import continued_fraction
from tqdm import tqdm

PICKLE_DIR = "/media/didi/TOSHIBA EXT/LIDAR/Pickles"

cwd = os.getcwd()
parent = os.path.abspath(os.path.join(cwd, os.pardir))

DATA_SAMPLE_DIR = parent + "/Data_sample"

# classes of point clouds:
CREATED = 0
UNCLASSIFIED = 1
GROUND = 2
LOW_VEGETATION = 3
MEDIUM_VEGETAION = 4
NATURAL_OBSTACLE = 5
HUMAN_STRUCTURES = 6
LOW_POINT = 7
MODEL_KEYPOINTS = 8
WATER = 9
RAIL = 10
ROAD_SURFACE = 11
OVERLAP_POINTS = 12
MEDIUM_RELIABILITY = 13
LOW_RELIABILITY = 14
POWER_LINE_SUPPORT_TOWER = 15
MAIN_POWER_LINE = 16
OTHER_POWER_LINE = 17
FIBER_OPTIC_CABLE = 18
NOT_RATED_OBJ_TBC = 19
NOT_RATED_OBJ_TBIG = 20
INCIDENTS = 21

DICT_NEW_LABELS = {
    CREATED : 0,
    UNCLASSIFIED : 0,
    LOW_POINT : 0, 
    MODEL_KEYPOINTS : 0,
    OVERLAP_POINTS : 0, 
    MEDIUM_RELIABILITY : 0,
    LOW_RELIABILITY : 0, 
    NOT_RATED_OBJ_TBC : 0, 
    NOT_RATED_OBJ_TBIG : 0, 
    RAIL : 0, # noise

    GROUND: 1,
    ROAD_SURFACE : 1, # ground

    LOW_VEGETATION : 2,
    MEDIUM_VEGETAION : 2, # vegetation

    NATURAL_OBSTACLE : 3,
    HUMAN_STRUCTURES : 3, 
    INCIDENTS : 3,# obstacles

    WATER : 4,
    POWER_LINE_SUPPORT_TOWER : 5,

    MAIN_POWER_LINE : 6,
    OTHER_POWER_LINE: 6,
    FIBER_OPTIC_CABLE : 6, # power lines
}

# useful constants
AVG_DIST_CLOSE_POINTS = 0.024479924860614104

"""
las.props:
['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 'edge_of_flight_line', 
'classification', 'synthetic', 'key_point', 'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'gps_time']
"""


def las_to_numpy(las) -> np.ndarray and np.ndarray:
    """
    Converts las file into a numpy file containing solely the xyz coords of the provided points\n
    Parameters
    ----------
        las: las object to be converted\n
    RET:
        pcnp: point cloud in numpy
        classes: classification of the points
    """
    pcnp = np.vstack((las.x, las.y, las.z)).transpose(
    )  # pcnp = point cloud in numpy
    # print(pcnp[:10])
    classes = np.array(las.classification)
    #print(classes[:10])

    # assert that all coordinates exist and are different from nan
    assert np.all(pcnp != np.nan)
    # assert that all points are classified
    assert len(pcnp) == len(classes)

    return pcnp, classes


def np_to_ply(xyz, save=False, filename="pcd.ply"):
    """
    Converts numpy ndarray into ply format in order to use open3D lib\n
    Parameters
    ----------
        `xyz` - numpy ndarray: np to be converted
        [`save` - boolean]: save ? saves the ply obj as filename in Data_Sample folder
        [`filename` - str]: name of the ply obj if save == True
        
    Returns
    -------
        pcd - ply: the corresponding ply obj
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if save:
        o3d.io.write_point_cloud(os.getcwd() + "/../Data_sample/" + filename, pcd)
        
    return pcd


def ply_to_np(pcd):
    return np.array(pcd.points)


def visualize_ply(pcd_load, window_name='Open3D'):
    """
    Plots the points clouds given as arg.

    Parameters
    ----------
    `pcd_load` - list of ply objs:
        N point clouds to be visualized
    """
   
    o3d.visualization.draw_geometries(pcd_load, window_name=window_name)


def get_tower_files(files_dirs = [DATA_SAMPLE_DIR], print_info=True):
    """
    Returns a list with all the .las files that have towers
    """
    tower_files = []
    tower_points = 0
    total_points = 0
    #files_dirs = ["media/didi/TOSHIBA\ EXT/LIDAR", "../Data_sample/"]
    

    for dir in files_dirs:
       
        os.chdir(dir)

        for las_file in os.listdir("."):
            twp, ttp = 0, 0
            filename = os.getcwd() + "/" + las_file
            if ".las" in filename:
                las = lp.read(filename)
            else:
                continue

            #print(f"las_file point format params:\n {list(las.point_format.dimension_names)}")

            xyz, classes = las_to_numpy(las)

            if np.any(classes == POWER_LINE_SUPPORT_TOWER):
                tower_files += [filename]
                twp = len(classes[classes == POWER_LINE_SUPPORT_TOWER])

            ttp = las.header.point_count
            if print_info:
                print(filename + "\n" +
                        f"file total points: {ttp} \n" +
                        f"file tower points: {twp}\nratio: {twp / ttp}\n")

            tower_points += twp
            total_points += ttp
        if print_info:
            print(f"\nnum of files with tower: {len(tower_files)}\n" +
                    f"total points: {total_points} \n" +
                    f"tower points: {tower_points}\nratio: {tower_points / total_points}\n")

    return tower_files


def get_las_from_dir(las_dirs = [DATA_SAMPLE_DIR], file_idxs='all', to_numpy=False):
    """
    Returns a list with all the .las files\\

    Parameters
    ----------

    `las_dirs` - List[String]:
        The directories where the .las files are locates

    `to_numpy` - bool:
        If true it converts the .las files to numpy format

    `file_idxs` - String | List[int]:
        if `file_idxs` == all : it loads all files if las_dir;
        if `file_idxs` == List[int] : returns files with said indexes;

    Returns
    -------

    List with .las files in original or numpy format
    """   

    files_count = 0 
    files = []    

    for dir in las_dirs:
       
        os.chdir(dir)

        list_dir = os.listdir(".")
        list_dir.sort()

        if file_idxs == 'all':
            file_idxs = np.arange(len(list_dir))

        for file_idx in file_idxs:
            filename = os.getcwd() + "/" + list_dir[file_idx]
            if ".las" in filename:
                las = lp.read(filename)
            else:
                continue

            #print(f"las_file point format params:\n {list(las.point_format.dimension_names)}")

            files_count += 1

            if to_numpy:
                xyz, classes = las_to_numpy(las)
                files.append([xyz, classes])
            else:
                files.append(las)

    if to_numpy:
        return [f[0] for f in files], [f[1] for f in files]

    return files

def merge_pcds(xyzs:List[np.ndarray], classes:List[np.ndarray]) -> Union[np.ndarray, np.ndarray]:

    merge = None 

    assert len(xyzs) == len(classes)

    for i in range(len(xyzs)):
        pcd = np.concatenate((xyzs[i], classes[i].reshape(-1, 1)), axis = 1)
        if merge is None:
            merge = pcd
        else:
            merge = np.concatenate((merge, pcd), axis=0)
    
    if merge is None:
        return None, None
    
    return merge[:, :-1], merge[:, -1]


def describe_data(X, y=None, col=['x', 'y', 'z'], print=True):
    """
    Describes the data X and returns the corresponding dataframe.\n
    Parameters
    ---------- 
        X:  ndarray to be described\n
        y:  ndarray with the classes of X \n
        col: names of the columns of the df to be returned (excluding y name)
    RET:
        df: dataframe of the described data with y included if y is not None
    PRE:
        len(X) == len(y) 
    """
    df = pd.DataFrame(X, columns=col)
    if y is not None:
        df["class"] = y
    if print:
        print(df.describe())
    return df


def normalize_xyz(data:np.ndarray):
    """
    Normalizes the data\n

    Parameters
    ----------
        `data`: ndarray with the data to be scaled

    Returns
    -------
        `scaler`: the scaler of the data
        `scaled_xyz`: the scaled data
    """
    scaler = MinMaxScaler()
    xyz_shape = data.shape
    scaled_xyz = scaler.fit_transform(data.reshape(-1, data.shape[-1]))
    return scaler, scaled_xyz.reshape(xyz_shape)


def xyz_centroid(xyz:np.ndarray) -> np.ndarray:

    # z_min, z_max = np.min(xyz, axis=0)[-1], np.max(xyz, axis=0)[-1]

    # cuts = 20

    # lin = np.linspace(z_min, z_max, num=cuts)

    # meds = np.zeros((cuts - 1, 3))

    # for i in range(cuts - 1):
    #     meds[i] = np.median(xyz[(xyz[:, -1] >= lin[i]) & (xyz[:, -1] <= lin[i+1])], axis=0)
    # print(meds)

    return np.median(xyz, axis=0)


def voxelize_ply(pcd_ply, voxelgrid_dims=(64, 64, 64), voxel_dims=None, plot=True):
    """
    Voxelizes the point cloud passed as argument\n

    Parameters
    ----------
    pcd_ply - ply obj:
        point cloud in open3d format
    voxel_dims - int tuple:
        the voxel dimensions to be used in voxelization
    plot - bool:
        if plot then plots voxelization 

    Returns
    -------
    pynt - pyntCloud instance:
        object that represents the point cloud passed as argument.\\
        contains the voxelization in pynt.structures list.
    """
    pynt = PyntCloud.from_instance("open3d", pcd_ply)
    if voxel_dims is None:
        x, y, z = voxelgrid_dims
        voxelgrid_id = pynt.add_structure("voxelgrid", n_x=x, n_y=y, n_z=z)
    else:
        x, y, z = voxel_dims
        voxelgrid_id = pynt.add_structure(
            "voxelgrid", size_x=x, size_y=y, size_z=z)
    voxel_grid = pynt.structures[voxelgrid_id]
    if plot:
        print(f"Voxel grid dim: {voxel_grid.x_y_z}\nVoxel Shape: {voxel_grid.shape}")
        voxel_grid.plot(d=3, mode="density", cmap="hsv")
    return pynt, voxelgrid_id


def downsampling(pcd, classes, samp_per=0.5):
    """
    Down samples the given point cloud
    by sampling some points by a certain percentage.
    This way the function maintains the point classes.
    \n
    Parameters
    ----------
    pcd - ply obj: 
        point cloud data in ply format
    samp_per - float: 
        sample percentage. How much of the point cloud will we roughly retain

    Returns
    -------
    downpdc - np 3darray:
        xyz point cloud down sampled
    classes - np 1darray:
        corresponding classes
    """

    cloud, id = voxelize_ply(pcd, voxelgrid_dims=(64, 64, 64), plot=False)
    grid = cloud.structures[id]

    voxels = dict()
    xyz = ply_to_np(pcd)
    for i, point in enumerate(xyz):
        idx = grid.voxel_n[i]
        vox = voxels.get(idx, list())
        vox.append(int(i))
        voxels[idx] = vox

    used_voxels = np.fromiter(voxels.keys(), dtype=int)
    sampling = np.zeros(len(xyz))
    counter = 0
    for vox in used_voxels:
        npvox = np.array(voxels[vox])
        selected = np.random.rand(len(npvox))
        sample = npvox[selected <= samp_per]
        end = counter + len(sample)
        sampling[counter:end] = sample
        counter = end
    sampling = sampling[:counter].astype(int)
    assert len(sampling) == counter

    return xyz[sampling], classes[sampling]


def downsampling_relative_height(xyz, classes, sampling_per=0.8):
    """
    Downsamples a point cloud according to the height of the point cloud, 
    the lower the height, the higher the downsample applied. 

    Parameters
    ----------
    xyz - (N, 3) numpy array:
        point cloud in numpy format

    classes - (1, N) numpy array:
        classes of the point cloud

    sampling_per - float \in [0, 1]:
        percentage to downsample the data with
    
    Returns
    -------
        (down_xyz, down_classes)
        Downsampled point cloud in numpy format (M, 3), w. M <= N
        Respective classes
    """

    pcd = np_to_ply(xyz)
    cloud, id = voxelize_ply(pcd, voxelgrid_dims=(64, 64, 64), plot=False)
    grid = cloud.structures[id]

    voxels = dict()
    for i, point in enumerate(xyz):
        idx = grid.voxel_z[i]
        vox = voxels.get(idx, list())
        vox.append(int(i))
        voxels[idx] = vox #voxels are aggregated according to their z coord

    sampling = np.zeros(len(xyz), dtype=np.int) #the downsample will have at most len(xyz) samples
    counter = 0

    for z in voxels:
        height_idxs = np.array(voxels[z]) # idx of points at height z

        normalized_z = z / (grid.x_y_z[2] - 1)
        selected = np.random.rand(len(height_idxs)) # uniformly random number for each idx 

        sample = height_idxs[selected <= sampling_per*(1 - normalized_z)] 
        end = counter + len(sample)
        sampling[counter:end] = sample
        counter = end

    sampling = sampling[:counter].astype(int)
    assert len(sampling) == counter

    return xyz[sampling], classes[sampling]


def avg_dist_points(xyz, accuracy, num_dists):
    """
    calculates the average distance between points 
    considering accuracy% points of the point cloud.
    E.g., if acc = 0.5, we consider half of the points in xyz
    \n

    Parameters
    ----------
    xyz - ndarray:
        point cloud data in numpy format
    accuracy - float:
        percentage of how much data to consider for this avg
    num_dists - int:
        number of distances to use in the average
    """

    num_iter = round(len(xyz) * accuracy)
    xyz_view = xyz[:num_iter]
    dists = []
    avgs = []
    for point in xyz_view:
        for neigh in xyz_view:
            euc = euclidean_distance(point, neigh)
            if euc > 0:
                dists.append(euc)
        avgs.append(np.mean(np.sort(dists)[:num_dists]))

    return np.mean(avgs)


def select_object(xyz, classes, obj_class):
    """
    Performs object selection on the point cloud of the point with the designated class.\n
    Parameters
    ----------
        xyz - ndarray: numpy point cloud
        classes - ndarray: classes of all points in xyz
        obj_class - ndarray: 1 or more classes of desired objects
    RET:
        ply point cloud with the selected objects;
        their corresponding classes
    """
    a = np.append(xyz, classes.reshape(-1, 1), axis=1)
    xyz_obj = a[np.isin(classes, obj_class)]
    return np_to_ply(xyz_obj[:, :-1]), a[:, -1]


def color_pointcloud(pcd, classes, load=False, pck_name="class_colors", class_color=None):
    """
    Colors the given point cloud.\n

    Parameters
    ----------
    `pcd` - PointCloud : 
        point cloud data to be colored with n points;
    `classes` - ndarray: 
        classification of the points in pcd; 
    `load` - boolean: 
        load ? loads pickle with assigned colors for each pixel;
    `pck_name` - str: 
        filename for the pickle to be used it load = True;
    `class_colors` - (22, 3) array: 
        for if specific class colors want to be used;

    Pre-Conds
    ---------
        class_color.shape == (22, 3);
        np_colors.shape == (len(classes), 3) -> pickle file shape;

    Returns
    -------
    np_colors - np.ndarray:
        The colors for each class in format (N, 3)
    """
    if load:
        np_colors = load_pickle(pck_name)
    else:
        unique_classes = np.unique(classes)

        if class_color is None:
            # define random colors for each class & normalize them
            class_color = np.random.choice(
                256, size=(len(unique_classes), 3)) / 255

        np_colors = np.empty((len(classes), 3))
        for i, c in enumerate(classes):
            np_colors[i, :] = class_color[np.where(unique_classes == c)[0][0]]
        """
        # map each class to the corresponding generated color
        np_colors = np.array(
            [class_color[np.where(np.isclose(unique_classes, x))[0][0]] for x in classes])
        """
        assert np_colors.shape == (len(classes), 3)

    # assign colors to point cloud according to their class
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    return np_colors


def dbscan(pcd, eps, min_points, visual=True):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=visual))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(
        labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # -1 means noise
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    if visual:
        o3d.visualization.draw_geometries([pcd])


def save_pickle(data, pick_path):
    with open(pick_path, 'wb') as handle:
        cloudpickle.dump(data, handle)


def load_pickle(pick_path):
    with open(pick_path, 'rb') as handle:
        data = cloudpickle.load(handle)
    return data


def euclidean_distance(x:np.ndarray, y:np.ndarray, axis=None):
    assert x.shape == y.shape
    return np.linalg.norm(x - y, axis=axis)


def extract_towers(pcd_towers, eps=10, min_points=300, visual=False) -> List[np.ndarray]:
    """
    Extracts the points of each individual tower (or structure whose instances can be segregated using DBSCAN).\n
    Parameters
    ----------
        `pcd_towers` - pcd: 
            point cloud containing SOLELY the intended structures to be extracted
        `eps` - int: 
            neigh distance for DBSCAN algorithm
        `min_points` - int: 
            min number of neigh for DBSCAN algorithm
        (these 2 params are optimized to segregate towers)

    Returns
    -------
        list with N numpy arrays, each numpy contains an instance (xyz coords) of the intended structure.
    """
    dbscan(pcd_towers, eps, min_points, visual)

    pcd_tower_points = np.array(pcd_towers.points)
    pcd_tower_colors = np.array(pcd_towers.colors)

    # each color represents a cluster from the DBSCAN
    tower_colors = np.unique(pcd_tower_colors, axis=0)
    
    # black is reserved for noise in DBSCAN ;
    tower_colors = tower_colors[(tower_colors != [0, 0, 0]).any(axis=1)]
    #assert len(pcd_tower_colors) == len(pcd_tower_points)

    if len(tower_colors) == 0: #no clusters from DBSCAN
        return []

    tower_xyz_rgb = np.append(pcd_tower_points, pcd_tower_colors, axis=1)
    # assert tower_xyz_rgb.shape == (len(pcd_tower_points), 6)  # x, y, z, r, g, b
    df_tower = describe_data(tower_xyz_rgb, col=['x', 'y', 'z', 'r', 'g', 'b'], print=False)
    group_rgb = df_tower.groupby(['r', 'g', 'b'])

    # towers will contain numpys with the coords of each individual tower in the .las file
    towers = [np.array(group_rgb['x', 'y', 'z'].get_group(tuple(color)))
              for color in tower_colors]
    # there are as many towers at the end as clusters in the DBSCAN
    # assert len(towers) == len(tower_colors)

    return towers


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def crop_tower_radius(xyz, classes, xyz_tower, radius=0):
    """
    Selects points from the original point cloud around a given tower considering a given radius\n

    Parameters
    ----------
        xyz - numpy 3darray: 
            xyz coords of the original point cloud
        classes - numpy 1darray:
            classes of the the xyz points
        xyz_tower - numpy array: 
            xyz coords of the tower to be cropped
        radius - int: 
            radius around the tower

    Returns
    -------
        rad - numpy 3darray: 
            points of the original point cloud around the tower at radius distance
        c - numpy 1darray:
            rads corresponding classes
    """
    if radius == 0:
        # radius = height of the tower
        radius = np.max(xyz_tower[:, 2]) - np.min(xyz_tower[:, 2])

    baricenter = np.mean(xyz_tower, axis=0)
    a = np.append(xyz, classes.reshape(-1, 1), axis=1)
    rad = a[np.sum(np.power((a[:, :-2] - baricenter[:-1]), 2),
                   axis=1) <= radius*radius]

    return rad[:, :-1], rad[:, -1].astype(int)


def crop_two_towers(xyz, classes, xyz_tower1, xyz_tower2):
    """
    Selects the point cloud area between the two given towers.\n
    Parameters
    ----------
    `xyz` - numpy array: 
        xyz coords of the original point cloud
    `classes` - numpy 1darray:
        classes of the the xyz points
    `xyz_tower1` -  numpy array: 
        xyz coords of tower1
    `xyz_tower2` - numpy array: 
        xyz coords of tower2\n

    Returns
    -------
    `ret` - numpy 2darray: 
        xyz coords of the original point cloud that are between the given towers.
    `c` - numpy 1darray:
        rets corresponding classes
    """
    # m1 = np.min(xyz_tower1, axis=0)
    # m2 = np.min(xyz_tower2, axis=0)

    # if m1[1] > m2[1]:
    #     t1 = xyz_tower2
    #     t2 = xyz_tower1
    # else:
    #     t1 = xyz_tower1
    #     t2 = xyz_tower2

    # ensure that t1 precede t2 in space; we disregard the z coord
    tt = np.concatenate((xyz_tower1, xyz_tower2))
    min1 = np.min(tt, axis=0)
    max2 = np.max(tt, axis=0)
    
    a = np.append(xyz, classes.reshape(-1, 1), axis=1)
    a = a[((min1 <= a[:, :-1]) & (a[:, :-1] <= max2))[:, :-1].all(axis=1)]  # we disregard the z coord

    return a[:, :-1], a[:, -1].astype(int)


def crop_ground_samples(xyz, classes, max_tower_h=40):


    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    samples = []
    step = int((xyz_max[0] - xyz_min[0]) / 100)

    for x in np.linspace(xyz_min[0], xyz_max[0], step):

        a = np.append(xyz, classes.reshape(-1, 1), axis=1)
        rad = a[np.logical_and(a[:, 0] >= x, a[:, 0] <= x + step)]

        if len(rad) > 300 and len(np.unique(rad[:, -1])) >= 2:
            # visualize_ply([np_to_ply(rad[:, :-1])])

            if not POWER_LINE_SUPPORT_TOWER in rad[:, -1].astype(int):
                rad[:, -1] = rad[:, -1].astype(int)
                samples.append(np.copy(rad))
               
    return samples


def crop_two_towers_samples(xyz:np.ndarray, classes:np.ndarray) -> List[np.ndarray]:

    pcd_tower, _ = select_object(xyz, classes, [POWER_LINE_SUPPORT_TOWER])
    towers = extract_towers(pcd_tower, visual=False)

    if len(towers) == 1:
        return []

    samples = []

    avg_points = np.array([np.mean(tower, axis=0) for tower in towers])

    for i in range(len(towers)):
        eucs = np.array([euclidean_distance(avg_points[i], avg_points[j]) for j in range(len(towers))])
        # idx is the index of the closest tower to towers[i]
        idx = np.argmin(eucs[eucs > 0]) # eucs == 0 is the distance with itself
        if idx >= i:
            idx += 1

        crop_2, crop_2_cl = crop_two_towers(xyz, classes, towers[i], towers[idx])
        if len(crop_2) == 0:
            continue
        tower_2 = np.append(crop_2, crop_2_cl.reshape(-1, 1), axis=1)
        cropt1, cropt1_cl = crop_tower_radius(xyz, classes, towers[i])
        t1 = np.append(cropt1, cropt1_cl.reshape(-1, 1), axis=1)
        cropt2, cropt2_cl = crop_tower_radius(xyz, classes, towers[idx])
        t2 = np.append(cropt2, cropt2_cl.reshape(-1, 1), axis=1)
        final_crop = np.concatenate((tower_2, t1, t2))

        samples.append(final_crop)

        # pcd = np_to_ply(final_crop[:, :3])
        # cl = final_crop[:, 3]
        # color_pointcloud(pcd, cl)
        # visualize_ply([pcd])

    return samples



def crop_tower_samples(xyz:np.ndarray, classes:np.ndarray, obj_class=[POWER_LINE_SUPPORT_TOWER]) -> List[np.ndarray]:

    pcd_tower, _ = select_object(xyz, classes, obj_class)
    towers = extract_towers(pcd_tower, visual=False)

    samples = []

    for tower in towers:
        crop, crop_classes = crop_tower_radius(xyz, classes, tower, radius=15)
        tower_section = np.append(crop, crop_classes.reshape(-1, 1), axis=1)
        samples.append(tower_section)

    return samples


def crop_at_locations(xyz:np.ndarray, coords:np.ndarray, radius:float=0, classes:np.ndarray=None) -> List[np.ndarray]:
    
    samples = []

    class_idx = 0
    if classes is not None:
        xyz = np.append(xyz, classes.reshape(-1, 1), axis=1)
        class_idx = -1

    if radius == 0:
        # radius = height of the tower
        radius = np.max(xyz[:, 2]) - np.min(xyz[:, 2])

    for c in coords:
        baricenter = c

        rad = xyz[np.sum(np.power((xyz[:, :-1+class_idx] - baricenter[:-1]), 2), axis=1) <= radius*radius]
        samples.append(rad)
        

    return samples




# %%
if __name__ == "__main__":

    LAS_FILES = "/home/didi/VSCode/lidar_thesis/Data_sample"

    NPY_FILES = "/home/didi/VSCode/lidar_thesis/dataset/raw_dataset/samples"
    
    # get_tower_files([LAS_FILES])

    # convert_las_to_npy([DATA_SAMPLE_DIR, DATA_SAMPLE_DIR + "/npys"])

    #tower_files = get_tower_files()

    # for tower_file in tower_files:
    #xyz, classes = las_to_numpy(lp.read(tower_files[1]))

    for file in os.listdir(NPY_FILES):
        f_name = os.path.join(NPY_FILES, file)
        xyz = np.load(f_name, allow_pickle=True)
        pynt = np_to_ply(xyz[:, :-1])
        color_pointcloud(pynt, xyz[:, -1])
        visualize_ply([pynt])

        input("Press Enter to continue...")

    # %%
    tower_files = get_tower_files()

    for f in tower_files[1:]:
        xyz, classes = las_to_numpy(lp.read(f))
        #visualize_ply([np_to_ply(xyz)])
        crop_two_towers_samples(xyz, classes)
    # %%

    #######
    # This is to plot the density of the TS40K dataset for each label
    #######
    LAS_FILES = "/home/didi/VSCode/lidar_thesis/Data_sample"
    EXT_DIR = "/media/didi/TOSHIBA EXT/LIDAR/"
    bins = np.arange(22)
    freq_dict = dict([(b, 0) for b in bins])
    for dir in [LAS_FILES, EXT_DIR]:
        for f in tqdm(os.listdir(dir)):
            f = os.path.join(dir, f)
            if os.path.isfile(f) and '.las' in f:
                unique, counts = np.unique(las_to_numpy(lp.read(f))[1], return_counts=True)
                freqs = dict(zip(unique, counts))
                for fr in freqs:
                    freq_dict[fr] += freqs[fr]
            if np.random.randint(0, 10, 1)[0] % 10 == 0:
                pprint(freq_dict) 
            #classes.append(las_to_numpy(lp.read(f))[1])


    # classes = np.concatenate(classes)
    # xyz, classes = las_to_numpy(lp.read(tower_files[1]))
    # print(classes.shape)
    # unique, counts = np.unique(classes, return_counts=True)
    # freq_dict = dict(zip(unique, counts))

    # %%

    #######
    # This is to plot the density of the raw dataset for each label
    #######
    RAW_DATASET = "/home/didi/VSCode/lidar_thesis/dataset/raw_dataset/samples"
    bins = np.arange(22)
    freq_dict = dict([(b, 0) for b in bins])
    for dir in [RAW_DATASET]:
        for f in tqdm(os.listdir(dir)):
            f = os.path.join(dir, f)
            if os.path.isfile(f) and '.npy' in f:
                npy:np.ndarray = np.load(f)
                #print(npy.shape)
                unique, counts = np.unique(npy[:, -1], return_counts=True)
                freqs = dict(zip(unique, counts))
                for fr in freqs:
                    freq_dict[fr] += freqs[fr]
            if np.random.randint(0, 10, 1)[0] % 10 == 0:
                pprint(freq_dict)
    pprint(freq_dict) 

    # %%
    bins = list(freq_dict.keys())
    class_count = list(freq_dict.values())
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize =(16, 9))
    rects = ax.barh(bins, class_count, align='center')

    pprint(dict([(l, f"{c*100 / sum(class_count):.3f}") for l, c in freq_dict.items()]))

    # Remove axes splines
    for s in ['top', 'bottom','left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)

    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

    for rect in rects:
        width = rect.get_width()
        # plt.text(1.05*rect.get_width(), rect.get_y()+0.5*rect.get_height(),
        #          f'{np.round(int(width) / sum(class_count), decimals=3):.3f}',
        #          ha='center', va='center')
        prob = np.round(int(width)*100 / sum(class_count), decimals=3)
        text = f'{prob:.3f}%' if prob > 0 else f'{int(prob)}%'
        plt.annotate(text, # this is the text
                    (rect.get_width(), rect.get_y()+ 0.5*rect.get_height()), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(25,0), # distance from text to points (x,y)
                    ha='center', va='center') # horizontal alignment can be left, right or center
    print('hi')
    ax.set_yticks(bins)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Frequency')
    ax.set_xlim(right=np.max(class_count)*1.01)
    ax.set_title('Label Distribution')
    #plt.show()
    #fig.set_size_inches(10*2.54, 6*2.54)
    plt.tight_layout()
    plt.savefig(os.path.join("/home/didi/VSCode/lidar_thesis/EDA", "label_dist2.png"), dpi=200, facecolor='white', transparent=False)    
    # stat data on the point cloud
    # %%
    # df = describe_data(xyz, classes)
    # print(f"Missing values:\n{df.isnull().sum()}")

    pcd_tower, _ = select_object(xyz, classes, [POWER_LINE_SUPPORT_TOWER])
    pcd_power_lines, _ = select_object(xyz, classes, [MAIN_POWER_LINE])
    visualize_ply([pcd_tower])
    #visualize_ply(pcd.voxel_down_sample(voxel_size = 5))

    # %%
    """
    Visualize the point cloud coloured by their class.
    """
    pcd = np_to_ply(xyz)
    visualize_ply([pcd])
    #color_pointcloud(pcd, classes)
    visualize_ply([pcd])

    # %%
    """
    Our objective is to segregate the tower so that we can visualize them better and
    perform stat analysis on the point clouds that composed them
    """
    towers = extract_towers(pcd_tower)
    crop, _ = crop_tower_radius(xyz, classes, towers[0])
    visualize_ply([np_to_ply(crop)])
    # %% 
    ground_sections = crop_ground_samples(xyz, classes)

    visualize_ply([np_to_ply(x) for x, y in ground_sections])



    # %%
    """
    Some simple stat analysis on the towers
    """
    avg_points = np.empty((len(towers), 3))
    tower_dims = np.empty((len(towers), 3))
    for i in range(len(towers)):
        avg_points[i] = np.mean(towers[i], axis=0)
        tower_dims[i] = np.max(towers[i], axis=0) - np.min(towers[i], axis=0)

    angles = []
    for i in range(len(towers)):
        bar = np.copy(avg_points[i])
        # change the z coord the the highest point of the tower
        bar[2] = np.max(towers[i], axis=0)[2]
        base_point = np.min(towers[i], axis=0)
        base_z = np.copy(base_point)
        base_z[2] = bar[2]
        angle = round(angle_between(bar - base_point, base_z - base_point), 3)
        angles.append(angle)

    for i in range(len(towers)):
        eucs = np.array([euclidean_distance(avg_points[i], avg_points[j])
                        for j in range(len(towers))])
        print(f"Tower {i}:\n\tNum points in tower: {len(towers[i])}\n\tTower mean point: {avg_points[i]}\n\t" +
              f"Min/Max euclidean distance to other towers: {round(np.min(eucs[eucs > 0]), 3), round(np.max(eucs[eucs > 0]), 3)}\n" +
              f"\tTower dims: {tower_dims[i]}\n\tAngle between the base and the top point of the tower: {angles[i]} degrees\n")

    print(
        f"--- Overall Stats ---\n\tAverage towers dimensions: {np.mean(tower_dims, axis = 0)}")
    lens = np.array([len(tow) for tow in towers])
    print(f"\tMin/Max tower point count: {np.min(lens)}, {np.max(lens)}")

    # %%

    for i in range(len(towers)):
        eucs = np.array([euclidean_distance(avg_points[i], avg_points[j])
                        for j in range(len(towers))])
        idx = np.argmin(eucs[eucs > 0])
        if idx >= i:
            idx += 1
        visualize_ply(
            [np_to_ply(crop_two_towers(xyz, classes, towers[i], towers[idx]))])
        # np_to_ply(crop_tower_radius(xyz, towers[i])),
        # np_to_ply(crop_tower_radius(xyz, towers[idx]))])

    for tower in towers:
        visualize_ply([np_to_ply(crop_tower_radius(xyz, classes, tower))])

    # %%

    for tower in towers:
        visualize_ply([np_to_ply(tower)])
    visualize_ply([np_to_ply(tower) for tower in towers])

    # %%
    # build_data_samples([DATA_SAMPLE_DIR], DATA_SAMPLE_DIR + "/npys")

    # %%
    save_pickle(np.array(pcd.colors), "class_colors")
