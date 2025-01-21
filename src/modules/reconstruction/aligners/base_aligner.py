from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
import open3d as o3d

from modules.reconstruction.utilities.loki import LokiDataset
from modules.reconstruction.utilities.pointcloud_utils import PointCloudUtils
from modules.reconstruction.utilities.recon_3d_config import Reconstuction3DConfig


class BaseAligner(ABC):
    """
    Abstract base class for aligners. Provides common functionality for loading
    point clouds, transformations, and scenario data from a LOKI dataset.
    """

    def __init__(self, scenario_path: Optional[str], loki_csv_path: str) -> None:
        
        """
        Initialize the BaseAligner with scenario paths and dataset references.

        Args:
            scenario_path (str): Path to the scenario directory containing point cloud and odometry files.
            loki_csv_path (str): Path to the LOKI CSV file containing additional data.
        """
        self.scenario_path = scenario_path
        self.loki_csv_path = loki_csv_path
        self.max_frames = Reconstuction3DConfig.frames_max_threshold
        self.frame_step = Reconstuction3DConfig.frame_step
        self.loki = LokiDataset(scenario_path, loki_csv_path)

    @abstractmethod
    def align(self, *args, **kwargs):
        """
        Abstract method to process the alignment of point clouds.

        Derived classes should implement alignment logic based on 
        odometry or other data sources.
        """
        pass

    def get_transformation_matrix(self, odometry: Tuple[float, float, float, float, float, float]) -> np.ndarray:
        """
        Generate a 4x4 transformation matrix from odometry data.

        Args:
            odometry (Tuple[float, float, float, float, float, float]): 
                Odometry data in the form [x, y, z, roll, pitch, yaw].

        Returns:
            np.ndarray: The corresponding 4x4 transformation matrix.
        """
        return PointCloudUtils.get_transformation_matrix(odometry)

    def load_point_cloud(self, file_path: str) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from a given file path.

        Args:
            file_path (str): The path to the point cloud file (.ply, .pcd, etc.).

        Returns:
            o3d.geometry.PointCloud: The loaded point cloud object.
        """
        return PointCloudUtils.load_point_cloud(file_path)

    def save_point_cloud(self, file_path: str, pointcloud: o3d.geometry.PointCloud) -> None:
        """
        Save a point cloud to the specified file path.

        Args:
            file_path (str): The file path where the point cloud will be saved.
            pointcloud (o3d.geometry.PointCloud): The point cloud to save.
        """
        PointCloudUtils.save_point_cloud(pointcloud, file_path)
