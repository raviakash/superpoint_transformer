"""
---------------------------------------------------------------------------------
Copyright (c) HOMIRO metrology GmbH. All rights reserved.
---------------------------------------------------------------------------------
Datum: 28.02.2025
Help Functions for preprocessing of point clouds.

"""

from __future__ import annotations

import laspy
import open3d as o3d
import numpy as np
import numpy.typing as npt
import os
import CSF
import logging
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc
import pandas as pd
from tqdm import tqdm
import numba
from numba import njit, prange
from numba.typed import List
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class PointCloud:
    def __init__(self):
        """

        A python class to handle original uncropped point cloud along with trajectory,
        perform both circular and square crop based on trajectory and saving point clouds in .las formats.

        """
        self.points = None
        self.trajectory_points = None
        
    def load_from_folder(self, folder_path: str, trajectory_type: str = 'txt', trajectory_crop: bool = True) -> tuple[str, Optional[str]]:
        """
        Loads both main point cloud and trajectory by detecting appropriate files in a folder.
        Stores paths internally and calls readers.

        Parameters:
            folder_path (str): Path to the folder containing the files.
            trajectory_type (str): 'ply' or 'txt' to specify the trajectory file type.
            trajectory_crop (bool): If True, look for and load trajectory file. If False, skip trajectory file logic.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = os.listdir(folder_path)

        laz_files = [f for f in files if f.lower().endswith(('.las', '.laz'))]
        ply_files = [f for f in files if f.lower().endswith('.ply')]
        txt_files = [f for f in files if f.lower().endswith('.txt')]

        if len(laz_files) == 0:
            raise FileNotFoundError("No .las or .laz file found in the folder.")
        if len(laz_files) > 1:
            raise RuntimeError("Multiple .las/.laz files found. Please ensure only one is present.")

        self.main_path = os.path.join(folder_path, laz_files[0])
        self.trajectory_path = None

        self.read_point_cloud(self.main_path)

        if trajectory_crop:
            if trajectory_type == 'ply':
                if len(ply_files) == 0:
                    raise FileNotFoundError("No .ply file found in the folder.")
                if len(ply_files) > 1:
                    raise RuntimeError("Multiple .ply files found. Please ensure only one is present.")
                trajectory_file = ply_files[0]
            elif trajectory_type == 'txt':
                if len(txt_files) == 0:
                    raise FileNotFoundError("No .txt file found in the folder.")
                if len(txt_files) > 1:
                    raise RuntimeError("Multiple .txt files found. Please ensure only one is present.")
                trajectory_file = txt_files[0]
            else:
                raise ValueError("trajectory_type must be either 'ply' or 'txt'")

            self.trajectory_path = os.path.join(folder_path, trajectory_file)
            self.read_trajectory()
            logger.info("Loaded point cloud from: %s", self.main_path)
            logger.info("Loaded trajectory from: %s", self.trajectory_path)
            return self.main_path, self.trajectory_path
        else:
            logger.info("Loaded point cloud from: %s", self.main_path)
            return self.main_path, None
    def read_point_cloud(self, file_path: str) -> npt.NDArray[np.float64]:
        """
        Reads a point cloud from a .las/.laz or .ply file and returns it as a NumPy array.

        Parameters:
        file_path (str): Path to the point cloud file.

        Returns:
        numpy.ndarray: Nx3 array of point coordinates.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.las', '.laz']:
            t0 = time.perf_counter()
            las = laspy.read(file_path)
            logger.debug("time laspy.read: %.6f in PointCloud.read_point_cloud", time.perf_counter() - t0)
            self.points = np.vstack((las.x, las.y, las.z)).T  # Convert to Nx3 NumPy array
        elif ext == '.ply':
            original_points = o3d.io.read_point_cloud(file_path)
            self.points = np.asarray(original_points.points)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return self.points
    
    def read_trajectory(self) -> npt.NDArray[np.float64]:
        """
        Loads the trajectory either from a .ply point cloud or a .txt file with structured columns.
        """
        if self.trajectory_path.lower().endswith('.ply'):
            # Read .ply trajectory using Open3D
            pcd = o3d.io.read_point_cloud(self.trajectory_path)
            self.trajectory_points = np.asarray(pcd.points)
            if self.trajectory_points.size == 0:
                raise RuntimeError(f"No points found in {self.trajectory_path}.")
            return self.trajectory_points

        elif self.trajectory_path.lower().endswith('.txt'):
            # Define expected column names
            columns = [
                'world_time', 'x', 'y', 'z',
                'q0', 'q1', 'q2', 'q3',
                'r', 'g', 'b',
                'nx', 'ny', 'nz',
                'roll', 'pitch', 'yaw'
            ]
            # Read trajectory data, skipping the header row
            df = pd.read_csv(self.trajectory_path, sep=r'\s+', names=columns, header=None, skiprows=1)

            if df.empty:
                raise RuntimeError(f"No data found in {self.trajectory_path}.")

            self.trajectory_points = df[['x', 'y', 'z']].values
            return self.trajectory_points

        else:
            raise ValueError(f"Unsupported trajectory file type: {self.trajectory_path}")    
    
    def get_trajectory_center(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Computes the geometric center (centroid) of trajectory data from either a .ply or .txt file.

        Returns:
        - tuple:
            - numpy.ndarray of shape (N, 3): trajectory points (x, y, z)
            - numpy.ndarray of shape (3,): the centroid (x, y, z)
        
        Raises:
        - RuntimeError: If no valid points are found.
        - ValueError: If the trajectory file type is unsupported.
        """
        if self.trajectory_path.lower().endswith('.ply'):
            # Read trajectory from .ply using Open3D
            pcd = o3d.io.read_point_cloud(self.trajectory_path)
            self.trajectory_points = np.asarray(pcd.points)

        elif self.trajectory_path.lower().endswith('.txt'):
            # Define expected column names
            columns = [
                'world_time', 'x', 'y', 'z',
                'q0', 'q1', 'q2', 'q3',
                'r', 'g', 'b',
                'nx', 'ny', 'nz',
                'roll', 'pitch', 'yaw'
            ]
            # Read .txt trajectory
            df = pd.read_csv(self.trajectory_path, sep=r'\s+', names=columns, header=None, skiprows=1)
            if df.empty:
                raise RuntimeError(f"No data found in {self.trajectory_path}. Cannot compute center.")
            
            # Use only first 1500 rows
            limited_df = df.iloc[:1500]
            self.trajectory_points = limited_df[['x', 'y', 'z']].values

        else:
            raise ValueError(f"Unsupported trajectory file format: {self.trajectory_path}")

        # Validate and compute centroid
        if self.trajectory_points.size == 0:
            raise RuntimeError(f"No points found in {self.trajectory_path}. Cannot compute center.")

        center_point = np.mean(self.trajectory_points, axis=0)
        logger.debug("Trajectory center computed: %s", center_point)

        return self.trajectory_points, center_point


    def save_point_cloud(self, point_cloud: npt.NDArray, save_path: str, format_choice: str) -> None:
        """
        Save the point cloud in the specified format (PLY, LAS, or LAZ).

        Args:
        - point_cloud: The point cloud data to save.
        - save_path: Path where the point cloud will be saved.
        - format_choice: The format to save the point cloud ('ply', 'las', or 'laz').
        """
        if format_choice == 'ply':
            # Save as PLY using open3d
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(point_cloud)
            o3d.io.write_point_cloud(save_path + '.ply', pcd_filtered)
            logger.info("Point cloud saved to: %s.ply", save_path)

        elif format_choice in ['las', 'laz']:
            # Convert the point cloud to a numpy array
            points = np.asarray(point_cloud)

            # Create a new LAS/LAZ file
            las = laspy.create(point_format=3, file_version='1.2')

            # Assign the x, y, z coordinates to the LAS file
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]

            # Save the LAS or LAZ file
            ext = '.laz' if format_choice == 'laz' else '.las'
            las.write(save_path + ext)
            logger.info("Point cloud saved to: %s%s", save_path, ext)

        elif format_choice == 'xyz':
            # Convert the NumPy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)

            # Save as .xyz
            o3d.io.write_point_cloud(save_path + '.xyz', pcd, write_ascii=True)
            logger.info("Point cloud saved to: %s.xyz", save_path)

        else:
            logger.error("Unsupported format '%s'. Use 'ply', 'las', 'laz', or 'xyz'.", format_choice)

    def crop_pointcloud_center(self, crop_size: float = 20) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Crops the loaded point cloud around its center and returns cropped points.

        Parameters:
        - crop_size (float): Size of the square crop region (in the same units as the point cloud).

        Returns:
        - tuple:
        - square_crop (numpy.ndarray): Nx3 array of point coordinates in the square region.
        - circle_crop (numpy.ndarray): Mx3 array of point coordinates in the circular region.
        """
        if self.points is None:
            raise ValueError("No point cloud loaded. Use read_point_cloud() first.")

        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]

        # Determine center of the point cloud
        #center_x, center_y = np.mean(x), np.mean(y)
        
        # Geometric center (Midpoint of the bounding box)
        center_x = (np.max(x) + np.min(x)) / 2
        center_y = (np.max(y) + np.min(y)) / 2
        half_crop = crop_size / 2

        # Define crop boundaries
        crop_xmin = center_x - half_crop
        crop_xmax = center_x + half_crop
        crop_ymin = center_y - half_crop
        crop_ymax = center_y + half_crop

        # Apply crop mask
        square_mask = (x >= crop_xmin) & (x <= crop_xmax) & (y >= crop_ymin) & (y <= crop_ymax)
        
        # Circular Mask
        radius_squared = half_crop ** 2
        dist_squared = (x - center_x) ** 2 + (y - center_y) ** 2
        circle_mask = dist_squared <= radius_squared
        
        # Apply Masks
        square_crop = np.vstack((x[square_mask], y[square_mask], z[square_mask])).T
        circle_crop = np.vstack((x[circle_mask], y[circle_mask], z[circle_mask])).T

        if square_crop.shape[0] == 0 and circle_crop.shape[0] == 0:
            logger.warning("No points found in the crop area.")
            
        return square_crop, circle_crop

    def crop_pointcloud_trajectory(self, crop_size: float = 20, center_point: Optional[npt.NDArray[np.float64]] = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Crops the loaded point cloud around a given center point or the point cloud's center.

        Parameters:
        - crop_size (float): Size of the square crop region (in the same units as the point cloud).
        - center_point (tuple or np.ndarray, optional): (x, y) coordinates to center the crop. 
        If None, the center is calculated from the point cloud.

        Returns:
        - tuple:
            - square_crop (numpy.ndarray): Nx3 array of point coordinates in the square region.
            - circle_crop (numpy.ndarray): Mx3 array of point coordinates in the circular region.
        """
        if self.points is None:
            raise ValueError("No point cloud loaded. Use read_point_cloud() first.")

        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]

        # Use provided center point or compute from point cloud
        if center_point is not None:
            center_x, center_y = center_point[0], center_point[1]
        else:
            center_x, center_y = np.mean(x), np.mean(y)

        # Radius of circle crop
        half_crop = crop_size / 2

        # Define square crop boundaries
        crop_xmin = center_x - half_crop
        crop_xmax = center_x + half_crop
        crop_ymin = center_y - half_crop
        crop_ymax = center_y + half_crop

        # Square crop mask
        square_mask = (x >= crop_xmin) & (x <= crop_xmax) & (y >= crop_ymin) & (y <= crop_ymax)

        # Circular crop mask (within radius from center)
        radius_squared = half_crop ** 2
        dist_squared = (x - center_x) ** 2 + (y - center_y) ** 2
        circle_mask = dist_squared <= radius_squared

        # Apply masks
        square_crop = np.vstack((x[square_mask], y[square_mask], z[square_mask])).T
        circle_crop = np.vstack((x[circle_mask], y[circle_mask], z[circle_mask])).T
        
        # Check for absence of points
        if square_crop.shape[0] == 0 and circle_crop.shape[0] == 0:
            logger.warning("No points found in the crop area.")

        return square_crop, circle_crop
    
    def save_cropped_pointclouds(self, crop_size: float = 20, save_dir: str = '.', user_center: Optional[npt.NDArray[np.float64]] = None, distance_threshold: float = 0.5, trajectory_crop: bool = True) -> None: 
        """
        Generates and saves square and circular cropped point clouds 
        centered on the trajectory, into separate subdirectories.

        Parameters:
        - crop_size (float): Size of the crop in the same units as the point cloud.
        - save_dir (str): Root directory where output subfolders will be created.
        - user_center (tuple, optional): User-specified center point (x, y).
        - distance_threshold (float): Threshold for deciding between trajectory center and user center.
        - trajectory_crop (bool): If True → crop around trajectory center, else crop around cloud center.
        """
        if self.main_path is None:
            raise ValueError("Original point cloud path not set. Use load_from_folder() first.")

        if trajectory_crop:
            # ---- Trajectory based cropping ----
            _, traj_center = self.get_trajectory_center()

            if user_center is not None:
                # Compute Euclidean distance between centers
                dist = np.linalg.norm(np.array(traj_center) - np.array(user_center))
                logger.debug("Distance between trajectory center and user center: %.2f", dist)

                # Decide which center to use based on distance threshold
                if dist <= distance_threshold:
                    center_to_use = traj_center
                else:
                    center_to_use = user_center
            else:
                # No user center passed, use trajectory center
                center_to_use = traj_center

            # Crop around trajectory center
            square_crop, circle_crop = self.crop_pointcloud_trajectory(crop_size=crop_size, center_point=center_to_use) 
            
        else:
            # ---- Cloud center based cropping ---- #
            square_crop, circle_crop = self.crop_pointcloud_center(crop_size=crop_size)
             
             
        # Extract filename prefix
        filename = os.path.splitext(os.path.basename(self.main_path))[0]

        # Define subdirectories
        square_dir = os.path.join(save_dir, 'square_crop')
        circle_dir = os.path.join(save_dir, 'circle_crop')

        # Create subdirectories if they don't exist
        os.makedirs(square_dir, exist_ok=True)
        os.makedirs(circle_dir, exist_ok=True)

        # Define full paths
        square_path = os.path.join(square_dir, f"{filename}_squarecrop")
        circle_path = os.path.join(circle_dir, f"{filename}_circularcrop")

        # Save cropped clouds
        self.save_point_cloud(square_crop, square_path, 'laz')
        self.save_point_cloud(circle_crop, circle_path, 'laz')

class Preprocessing:
    def __init__(self):
        self._points = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value

    @points.deleter
    def points(self):
        self._points = None


    def read_point_cloud(self, file_path: str) -> npt.NDArray[np.float32]:
        """
        Reads a point cloud from a .las/.laz or .ply file and returns it as a NumPy array.

        Parameters:
        file_path (str): Path to the point cloud file.

        Returns:
        numpy.ndarray: Nx3 array of point coordinates.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.las', '.laz']:
            t0 = time.perf_counter()
            las = laspy.read(file_path)
            logger.debug("time laspy.read: %.6f in Preprocessing.read_point_cloud", time.perf_counter() - t0)
            self.points = np.vstack((las.x, las.y, las.z)).T  # Convert to Nx3 NumPy array
            self.points = np.asarray(self.points, dtype=np.float32)
        elif ext == '.ply':
            original_points = o3d.io.read_point_cloud(file_path)
            self.points = np.asarray(original_points.points, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return self.points

    def downsample(self, voxel_size: float) -> npt.NDArray[np.float64]:
        """
        Downsamples the point cloud using a voxel grid and returns a NumPy array.

        Parameters:
        voxel_size (float): The size of the voxel grid.

        Returns:
        numpy.ndarray: Downsampled Nx3 array of point coordinates.
        """
        if self.points is None:
            raise ValueError("No point cloud loaded. Use read_point_cloud first.")

        # Convert to Open3D point cloud format
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(self.points)

        # Apply voxel downsampling
        downsampled_pc = o3d_pc.voxel_down_sample(voxel_size)

        # Convert back to NumPy array
        downsampled_points = np.asarray(downsampled_pc.points)
        self.points = downsampled_points  # Update stored point cloud

        return downsampled_points
    
    def downsample_and_trace(self, voxel_size: float = 0.05) -> tuple[npt.NDArray[np.float64], list, int]:
        """
        Downsample the loaded point cloud using a voxel grid filter and trace the mapping from original points to voxels.

        Parameters:
        - voxel_size: float, the size of the voxel grid

        Returns:
        - downpcd: open3d.geometry.PointCloud, the downsampled point cloud
        - idx: list of numpy arrays, each containing indices of original points within each voxel
        """
        if self.points is None:
            raise ValueError("No point cloud loaded. Use read_point_cloud() first.")

        # Convert to Open3D PointCloud if needed
        if isinstance(self.points, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
        elif isinstance(self.points, o3d.geometry.PointCloud):
            pcd = self.points
        else:
            raise TypeError("Unsupported point cloud type in self.points.")

        points = np.asarray(pcd.points)
        n_original_points = len(points)
        bound = np.max(np.abs(points)) + 100
        min_bound, max_bound = np.array([-bound, -bound, -bound]), np.array([bound, bound, bound])
        downpcd, _, idx = pcd.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound)
        downpcd_np = np.asarray(downpcd.points)
        return downpcd_np, idx, n_original_points

    def generate_dtm(self, bSloopSmooth: bool = True, cloth_resolution: float = 0.5, classify_threshold: float = 0.1, rigidness: int = 2) -> npt.NDArray[np.float64]:
        """
        Generates a Digital Terrain Model (DTM) using the Cloth Simulation Filter (CSF).

        Parameters:
        cloud : numpy.ndarray
            The point cloud. Matrix containing (x, y, z) coordinates of the points.
        bSloopSmooth : Boolean
            The resulting DTM will be smoothed. Defaults to True.
        cloth_resolution : float
            The resolution of the cloth grid. Defaults to 0.5.
        classify_threshold : float
            The height threshold used to classify the point cloud into ground and non-ground parts. Defaults to 0.1.
        rigidness : int
            CSF cloth rigidness. Lower value = more flexible cloth that drapes into steep terrain.
              1 — most flexible, recommended for steep / mountainous terrain (slopes > 30 deg)
              2 — moderate terrain (CSF default)
              3 — most rigid / stiff, recommended for flat terrain
            Defaults to 2.

        Returns:
        numpy.ndarray : DTM points (x, y, z).
        """
        csf = CSF.CSF()  # initialize the CSF

        csf.params.bSloopSmooth = bSloopSmooth
        csf.params.cloth_resolution = cloth_resolution
        csf.params.classify_threshold = classify_threshold
        csf.params.rigidness = rigidness

        csf.setPointCloud(self.points)  # pass the (x), (y), (z) list to CSF

        raw_nodes = csf.do_cloth_export()  # do actual filtering and export cloth
        cloth_nodes = np.reshape(np.array(raw_nodes), (-1, 3))

        return cloth_nodes
    
    def visualize_dtm(self, dtm_points: npt.NDArray[np.float64]) -> None:
        """
        Visualize the Digital Terrain Model (DTM) in 3D.

        Parameters:
        dtm_points (numpy.ndarray): Nx3 array of the DTM points (x, y, z).
        """
        x = dtm_points[:, 0]
        y = dtm_points[:, 1]
        z = dtm_points[:, 2]

        # Create a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color based on height (z-axis)
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=5)
        plt.colorbar(sc, label='Height (Z)')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of DTM')

        # Show the plot
        plt.show()

    def normalize_heights(self, dtm_points: npt.NDArray[np.float64], k_neighbours: int = 5) -> npt.NDArray[np.float64]:
        """
        Normalizes the heights of the point cloud based on the provided DTM.

        Parameters:
        dtm_points : numpy.ndarray
            Matrix containing (x, y, z) coordinates of the DTM points.
        k_neighbours : int
            Number of nearest DTM nodes used for inverse-distance-weighted interpolation.
            Higher values give a more stable ground estimate on steep or undulating terrain.
            Effective smoothing radius ≈ sqrt(k_neighbours) * cloth_resolution — re-tune if
            cloth_resolution changes significantly. Defaults to 5.

        Returns:
        numpy.ndarray : Normalized point cloud with adjusted z-values.
        """
        if self.points is None:
            raise ValueError("No point cloud loaded. Use read_point_cloud first.")

        # Query k nearest DTM nodes by XY only — Z is what we are computing.
        tree = KDTree(dtm_points[:, :2])
        d, idx_pt_mesh = tree.query(self.points[:, :2], k_neighbours, workers=-1)

        # Inverse-distance-squared weighting (Shepard p=2): closer nodes carry more weight.
        # np.maximum clamps exact-coincident distances to avoid 1/0; for d >> 1e-6 this is
        # exactly 1/d² with no bias introduced.
        weights = 1.0 / np.maximum(d, 1e-6) ** 2
        zs_diff = self.points[:, 2] - np.average(dtm_points[:, 2][idx_pt_mesh], weights=weights, axis=1)

        normalized_points = self.points.copy()
        normalized_points[:, 2] = zs_diff

        return normalized_points

    def random_subsample(self, point_cloud: npt.NDArray[np.float64], sample_fraction: float = 0.1) -> npt.NDArray[np.float64]:
        """
        Randomly subsample a fraction of the point cloud.

        Parameters:
        point_cloud : numpy.ndarray
            Matrix containing (x, y, z) coordinates of the point cloud.
        sample_fraction : float
            Fraction of points to sample (e.g., 0.1 for 10% of the points).

        Returns:
        numpy.ndarray : Subsampled point cloud.
        """
        num_points = len(point_cloud)
        num_samples = int(num_points * sample_fraction)
        indices = np.random.choice(num_points, num_samples, replace=False)
        subsampled_cloud = self.points[indices]
        return subsampled_cloud

    def remove_statistical_outliers(self, nb_neighbors: int = 10, std_ratio: float = 1.0, z_threshold: Optional[float] = 15,
                                    h_slice: Optional[float] = None, h_shift: Optional[float] = None) -> npt.NDArray[np.float64]:
        """
        Removes statistical outliers from a point cloud using Open3D's statistical outlier removal method.

        Parameters
        ----------
        point_cloud : numpy.ndarray
            Matrix containing (x, y, z) coordinates of the point cloud.
        nb_neighbors : int, optional
            Number of neighbors to consider for mean distance computation. Defaults to 10.
        std_ratio : float, optional
            Points with distances beyond (mean + std_ratio * std_dev) are considered outliers. Defaults to 1.0.
        z_threshold : float, optional
            Apply outlier removal only to points with z <= z_threshold. Points above remain unchanged.
        h_slice : float, optional
            Height of slices in meter.
            Apply outlier removal on individual z-slices to address different densities in
            different height. Could be used to preserve top portion of canopies. Defaults to None.
        h_shift : float, optional
            Height from one slice to another in meter.
            If outlier removal is applied on different z-slices this parameters determines about
            the shift beween to slices. Defaults to None.

        Returns
        -------
        numpy.ndarray : Filtered point cloud without statistical outliers (as a NumPy array).
        """
        # split the point cloud
        if z_threshold is None:
            z_threshold = self.points[:,2].max() + 1
        lower_points = self.points[self.points[:,2] <= z_threshold]
        upper_points = self.points[self.points[:,2] > z_threshold]

        # no points to filter?
        if lower_points.shape[0] == 0:
            return upper_points
        # filter only lower points

        # set-up z-slicing
        if h_slice is None:
            # no slicing => height of slice is maximum range of points
            h_slice = float(np.ptp(lower_points[:,2])) + 10 # add some number to ensure only one slice
            
        if h_shift is None:
            h_shift = h_slice
        elif h_shift < 1e-2:
            # avoid infinity loop
            h_shift = h_slice

        # set up slices for statistical outlier removal
        z_min = np.min(lower_points[:,2]) - 0.1 # add some space to catch definitely all points
        z_max = np.max(lower_points[:,2]) + 0.1
        z_bounds_slices = [(z0, z0 + h_slice) for z0 in np.arange(z_min, z_max, h_shift)]

        # add id to identify points which are part of multiple slices
        pts_with_id = np.hstack([lower_points, np.arange(lower_points.shape[0])[:,None]])

        # slice points and remove statistical outliers for each slice
        pcd = o3d.geometry.PointCloud()
        all_idx = []
        slice_idx = []
        for bound in z_bounds_slices:
            pts_slice = pts_with_id[(pts_with_id[:,2] >= bound[0]) * (pts_with_id[:,2] < bound[1])]
            pcd.points = o3d.utility.Vector3dVector(pts_slice[:,:3]) # only x-y-z
            _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            # use_idx += list(pts_slice[idx,3].astype(np.int32)) # for overlapping slices: take if at least one contains that pt -> use np.unique later
            all_idx.append(set(pts_slice[:,3].astype(np.int32))) # save for checking that points must be contained in two slices
            slice_idx.append(set(pts_slice[idx,3].astype(np.int32)))

        # take points from overlapping slices only if they were taken in both slices
        if h_shift < h_slice:
            N = len(z_bounds_slices)
            if N == 0:
                use_idx = set()
            elif N == 1:
                use_idx = slice_idx[0]
            else:
                all_idx_union = set().union(*all_idx)
                use_idx = set()
                for n in range(N):
                    # take all points which are not part in other slices
                    use_idx.update(slice_idx[n] - (all_idx_union - all_idx[n]))
                    
                    # for overlapping slices: take only those which were in both not filtered out
                    if n > 0:
                        use_idx.update(slice_idx[n] & slice_idx[n-1])
        else:
            use_idx = set().union(*slice_idx)

        # extract points to keep from those indices
        lower_points_cleaned = lower_points[sorted(use_idx)]
        filtered_points = np.vstack((lower_points_cleaned, upper_points))
        return filtered_points
    
    def noise_filter(self, kernel_radius: Optional[float] = None, n_sigma: float = 1.0, remove_isolated_points: bool = True, use_knn: bool = True, knn: int = 20, use_absolute_error: bool = False, absolute_error: float = 0.01, batch_size: int = 1_000_000) -> npt.NDArray[np.float64]:
        """
        Batched noise filtering on a point cloud based on local planarity.
        Returns filtered points as a NumPy array.
        """
        if self.points is None:
            raise ValueError("No point cloud loaded. Use read_point_cloud first.")

        points = np.asarray(self.points , dtype=np.float64)
        n_points = len(points)

        # Build KDTree once
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        # Precompute neighbors for all points
        neighbors_list = []
        for i in range(n_points):
            query = points[i]
            if use_knn:
                _, idxs, _ = kdtree.search_knn_vector_3d(query, knn + 1)
                idxs = [j for j in idxs if j != i][:knn]
            else:
                _, idxs, _ = kdtree.search_radius_vector_3d(query, kernel_radius)
                idxs = [j for j in idxs if j != i]
            neighbors_list.append(np.array(idxs, dtype=np.int32))
            
        # Convert to numba typed list
        neighbors_nb = List()
        for arr in neighbors_list:
            neighbors_nb.append(arr)

        # Numba-accelerated function for filtering
        @njit(parallel=True)
        def planar_filter(points, neighbors_list, n_sigma, use_absolute_error,
                        absolute_error, remove_isolated_points):
            n_points = points.shape[0]
            keep_mask = np.zeros(n_points, dtype=np.uint8)  # 0 = discard, 1 = keep

            for i in prange(n_points):
                idxs = neighbors_list[i]
                if idxs.size > 3:
                    neighbors = points[idxs]

                    # ---- compute centroid manually ----
                    centroid = np.zeros(3, dtype=np.float64)
                    for j in range(neighbors.shape[0]):
                        centroid[0] += neighbors[j, 0]
                        centroid[1] += neighbors[j, 1]
                        centroid[2] += neighbors[j, 2]
                    centroid /= neighbors.shape[0]

                    # ---- compute covariance manually ----
                    cov = np.zeros((3, 3), dtype=np.float64)
                    for j in range(neighbors.shape[0]):
                        diff0 = neighbors[j, 0] - centroid[0]
                        diff1 = neighbors[j, 1] - centroid[1]
                        diff2 = neighbors[j, 2] - centroid[2]
                        cov[0, 0] += diff0 * diff0
                        cov[0, 1] += diff0 * diff1
                        cov[0, 2] += diff0 * diff2
                        cov[1, 0] += diff1 * diff0
                        cov[1, 1] += diff1 * diff1
                        cov[1, 2] += diff1 * diff2
                        cov[2, 0] += diff2 * diff0
                        cov[2, 1] += diff2 * diff1
                        cov[2, 2] += diff2 * diff2
                    cov /= neighbors.shape[0]

                    # ---- eigendecomposition ----
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    normal = eigvecs[:, 0]
                    D = -np.dot(normal, centroid)

                    # ---- distances ----
                    dists_to_plane = np.empty(neighbors.shape[0], dtype=np.float64)
                    norm_n = np.linalg.norm(normal)
                    for j in range(neighbors.shape[0]):
                        dists_to_plane[j] = abs(np.dot(neighbors[j], normal) + D) / norm_n

                    if use_absolute_error:
                        max_dist = absolute_error
                    else:
                        # manual std
                        mean_d = 0.0
                        for j in range(dists_to_plane.shape[0]):
                            mean_d += dists_to_plane[j]
                        mean_d /= dists_to_plane.shape[0]
                        var_d = 0.0
                        for j in range(dists_to_plane.shape[0]):
                            diff = dists_to_plane[j] - mean_d
                            var_d += diff * diff
                        var_d /= dists_to_plane.shape[0]
                        std_d = var_d ** 0.5
                        max_dist = std_d * n_sigma

                    d = abs(np.dot(points[i], normal) + D) / norm_n
                    if d <= max_dist:
                        keep_mask[i] = 1
                else:
                    if not remove_isolated_points:
                        keep_mask[i] = 1

            # convert mask to indices
            keep_indices = np.nonzero(keep_mask)[0]
            return keep_indices

        
        # Apply the JIT-compiled filtering
        keep_indices = planar_filter(points, neighbors_list, n_sigma, use_absolute_error, absolute_error, remove_isolated_points)

        # Return filtered points as NumPy array
        filtered_points = points[keep_indices]
        return filtered_points

    def remove_ground_points(self, threshold: float = 0.2) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Removes ground points from a normalized point cloud, returns both the ground points and 
        the non-ground points, displays both the unfiltered and filtered point clouds using Matplotlib, 
        and optionally saves the filtered cloud.

        Parameters
        ----------
        normalized_cloud : numpy.ndarray
            Matrix containing (x, y, z) coordinates of the normalized point cloud.
        threshold : float, optional
            Height threshold to classify ground points. Defaults to 0.2 meters.

        Returns
        -------
        non_ground_points : numpy.ndarray
            Filtered point cloud without ground points.
        ground_points : numpy.ndarray
            Point cloud containing the ground points below the threshold.
        """
        # estimate the max_z value
        max_z = self.points[:,2].max()

        # modify the score threshold accordingly
        if max_z > 1000: # millimeters
            threshold_scaled = threshold * 1000
        elif max_z > 100: # centimeters
            threshold_scaled = threshold * 100
        else: # meters
            threshold_scaled = threshold

        # Filter out ground points
        ground_indices = self.points[:, 2] <= threshold_scaled
        ground_points = self.points[ground_indices]
        non_ground_points = self.points[~ground_indices]  # All points not classified as ground

        return non_ground_points, ground_points
    
    def csf_filter(self, cloth_resolution: float = 0.1, class_threshold: float = 2.0, bSloopSmooth: bool = True, rigidness: int = 2) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Filters ground and non-ground points using the Cloth Simulation Filter (CSF) directly on the loaded point cloud.

        Parameters:
        - cloth_resolution (float): The resolution of the cloth grid. Defaults to 0.1.
        - class_threshold (float): Threshold to classify ground vs non-ground. Defaults to 2.0.
        - bSloopSmooth (bool): Whether to smooth the output. Defaults to True.
        - rigidness (int): CSF cloth rigidness. 1=most flexible (steep/mountainous), 2=moderate, 3=most rigid (flat). Defaults to 2.

        Returns:
        - ground_points (np.ndarray): Nx3 array of ground points.
        - non_ground_points (np.ndarray): Nx3 array of non-ground points.
        """
        if self.points is None:
            raise ValueError("No point cloud loaded. Use read_point_cloud first.")

        csf = CSF.CSF()
        csf.setPointCloud(self.points)

        csf.params.bSloopSmooth = bSloopSmooth
        csf.params.cloth_resolution = cloth_resolution
        csf.params.class_threshold = class_threshold
        csf.params.rigidness = rigidness

        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)

        ground_indices = np.array(ground)
        non_ground_indices = np.array(non_ground)

        ground_points = self.points[ground_indices]
        non_ground_points = self.points[non_ground_indices]

        return non_ground_points, ground_points

    def save_point_cloud(self, normalized_point_cloud: npt.NDArray, save_path: str, format_choice: str) -> None:
        """
        Save the point cloud in the specified format (PLY or LAS or LAZ).

        Args:
        - normalized_point_cloud: The point cloud data to save.
        - save_path: Path where the point cloud will be saved.
        - format_choice: The format to save the point cloud ('ply' or 'las' or 'laz').
        """
        if format_choice == 'ply':
            # Save as PLY using open3d
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(normalized_point_cloud)
            o3d.io.write_point_cloud(save_path + '.ply', pcd_filtered)
            logger.info("Point cloud saved to: %s.ply", save_path)

        elif format_choice in ['las', 'laz']:
            # Convert the normalized point cloud to a numpy array
            points = np.asarray(normalized_point_cloud)

            # Create a new LAS file
            las = laspy.create(point_format=3, file_version='1.2')

            # Assign the x, y, z coordinates to the LAS file
            las.x = points[:, 0]
            las.y = points[:, 1]
            las.z = points[:, 2]

            # Save the LAS file
            ext = '.laz' if format_choice == 'laz' else '.las'
            las.write(save_path + ext)
            logger.info("Point cloud saved to: %s%s", save_path, ext)

        elif format_choice == 'xyz':
            # Convert the NumPy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(normalized_point_cloud)

            # Save as .xyz
            o3d.io.write_point_cloud(save_path + '.xyz', pcd, write_ascii=True)
            logger.info("Point cloud saved to: %s.xyz", save_path)

        else:
            logger.error("Unsupported format '%s'. Use 'ply', 'las', 'laz', or 'xyz'.", format_choice)

    def propagate_voxelized_to_original(self, voxel_points_transformed: npt.NDArray[np.float64], idx_mapping: list, n_original_points: int) -> npt.NDArray[np.float64]:
        """
        Propagates transformed voxelized points back to the original points.

        Args:
            voxel_points_transformed (np.ndarray): Mx3 array of transformed voxelized points.
            idx_mapping (list of lists or arrays): idx_mapping[i] contains indices of original points for voxel i.
            n_original_points (int): Number of original points.

        Returns:
            propagated_points (np.ndarray): Nx3 array of propagated points for the original cloud.
        """
        propagated_points = np.zeros((n_original_points, 3), dtype=voxel_points_transformed.dtype)
        n_voxels = len(voxel_points_transformed)
        
        for i, orig_indices in enumerate(idx_mapping):
            if i < n_voxels:  # Safety check to ensure we don't exceed voxel_points_transformed bounds
                propagated_points[orig_indices] = voxel_points_transformed[i]
            else:
                continue

        return propagated_points