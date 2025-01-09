from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import open3d as o3d

from .utils.pointcloud_utils import PointCloudUtils
from .utils.recon_3d_config import Reconstuction3DConfig
from .loki import LokiDataset


class BaseAligner(ABC):
    """
    Abstract base class for different types of processors.
    """

    def __init__(self, scenario_path: str, loki_csv_path: str) -> None:
        self.scenario_path = scenario_path
        self.loki_csv_path = loki_csv_path
        self.max_frames = Reconstuction3DConfig.frames_max_threshold
        self.frame_step = Reconstuction3DConfig.frame_step
        self.loki = LokiDataset(scenario_path, loki_csv_path)

    @abstractmethod
    def align(self, *args):
        """
        Abstract method to process the scenario.
        """
        pass

    def get_transformation_matrix(self, odometry: Tuple[float, float, float, float, float, float]) -> np.ndarray:
        return PointCloudUtils.get_transformation_matrix(odometry)

    def load_point_cloud(self, file_path: str) -> o3d.geometry.PointCloud:
        return PointCloudUtils.load_point_cloud(file_path)

    def save_point_cloud(self, file_path: str, pointcloud: o3d.geometry.PointCloud) -> None:
        PointCloudUtils.save_point_cloud(file_path, pointcloud)
