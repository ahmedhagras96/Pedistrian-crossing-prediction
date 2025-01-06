import os
import pandas as pd
import open3d as o3d
import re
from typing import Tuple, Optional, Any

from .utils.logger import Logger
from .utils.file_utils import FileUtils
from .utils.pointcloud_utils import PointCloudUtils


class LokiDataset:
    """
    Class to handle loading of LOKI dataset files including point clouds, odometry, and label3d data.

    Attributes:
        scenario_path (str): Path of the scenario.
        loki_csv_path (str): Base path to the loki.csv file.
        scenario_path (str): Path to the specific scenario directory.
        logger (logging.Logger): Logger instance for logging messages.
    """

    logger = Logger.get_logger(__name__)

    def __init__(self, scenario_path: str, loki_csv_path: str):
        """
        Initializes the LokiDataset with the specified scenario name and base LOKI path.

        Args:
            scenario_path (str): Path of the scenario (e.g., 'scenario_026').
            loki_csv_path (str): Base path to the loki.csv file.
                    """
        self.scenario_path = scenario_path
        self.scenario_id = re.search(r'scenario_(\d+)', os.path.basename(scenario_path), re.IGNORECASE).group(1)
        self.loki_csv_path = loki_csv_path

        LokiDataset.logger.info(f"Initialized {self.__class__.__name__} for scenario: {self.scenario_id}")

    def load_ply(self, frame_index: int) -> Optional[o3d.geometry.PointCloud]:
        """
        Loads the point cloud (.ply) file for the specified frame index.

        Args:
            frame_index (int): Index of the frame to load.

        Returns:
            Optional[o3d.geometry.PointCloud]: Loaded point cloud or None if loading fails.
        """
        ply_filename = f'pc_{frame_index:04d}.ply'
        ply_filepath = os.path.join(self.scenario_path, ply_filename)

        # LokiDataset.logger.info(f"Loading point cloud from: {ply_filepath}")

        if not FileUtils.file_exists(ply_filepath):
            LokiDataset.logger.error(f"Point cloud file does not exist: {ply_filepath}")
            return None

        try:
            pcd = PointCloudUtils.load_point_cloud(ply_filepath)
            # LokiDataset.logger.info(f"Successfully loaded point cloud for frame {frame_index}")
            return pcd
        except Exception as e:
            LokiDataset.logger.error(f"Failed to load point cloud for frame {frame_index}: {e}")
            return None

    def load_odometry(self, frame_index: int) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Loads the odometry data for the specified frame index.

        Args:
            frame_index (int): Index of the frame to load odometry data for.

        Returns:
            Optional[Tuple[float, float, float, float, float, float]]: Tuple containing
            (x, y, z, roll, pitch, yaw) or None if loading fails.
        """
        odom_filename = f'odom_{frame_index:04d}.txt'
        odom_filepath = os.path.join(self.scenario_path, odom_filename)

        # LokiDataset.logger.info(f"Loading odometry data from: {odom_filepath}")

        if not FileUtils.file_exists(odom_filepath):
            LokiDataset.logger.error(f"Odometry file does not exist: {odom_filepath}")
            return None

        try:
            with open(odom_filepath, 'r') as f:
                data = f.read().strip().split(',')
                if len(data) < 6:
                    LokiDataset.logger.error(f"Odometry file {odom_filepath} is malformed.")
                    return None
                odometry = tuple(map(float, data[:6]))
            # LokiDataset.logger.info(f"Successfully loaded odometry for frame {frame_index}")
        except Exception as e:
            LokiDataset.logger.error(f"Failed to load odometry for frame {frame_index}: {e}")
            return None

        return odometry

    def load_label3d(self, frame_index: int) -> Optional[pd.DataFrame]:
        """
        Loads the label 3d data for the specified frame index.

        Args:
            frame_index (int): Index of the frame to load label 3d data for.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing label 3d data or None if loading fails.
        """
        label3d_filename = f'label3d_{frame_index:04d}.txt'
        label3d_filepath = os.path.join(self.scenario_path, label3d_filename)

        # LokiDataset.logger.info(f"Loading label3d data from: {label3d_filepath}")

        if not FileUtils.file_exists(label3d_filepath):
            LokiDataset.logger.error(f"Label3d file does not exist: {label3d_filepath}")
            return None

        try:
            with open(label3d_filepath, 'r') as f:
                data = [line.strip().split(',') for line in f.readlines()]
            if not data:
                LokiDataset.logger.warning(f"Label3d file {label3d_filepath} is empty.")
                return pd.DataFrame()
            columns = [col.strip() for col in data[0]]
            df = pd.DataFrame(data[1:], columns=columns)
            # LokiDataset.logger.info(f"Successfully loaded label3d for frame {frame_index}")
            return df
        except Exception as e:
            LokiDataset.logger.error(f"Failed to load label3d for frame {frame_index}: {e}")
            return None

    def load_alignment_data(self, frame_index: int) -> Optional[Tuple[Any, Any, Any]]:
        """
        Loads the point cloud, odometry, and label 3d data for the specified frame index.

        Args:
            frame_index (int): Index of the frame to load data for.

        Returns:
            Optional[Dict[str, any]]: Dictionary containing 'point_cloud', 'odometry', and 'label3d' keys
            with their respective loaded data, or None if any loading fails.
        """
        LokiDataset.logger.info(f"Loading all data for frame {frame_index}")

        pcd = self.load_ply(frame_index)
        odometry = self.load_odometry(frame_index)
        label3d = self.load_label3d(frame_index)

        if pcd is None or odometry is None or label3d is None:
            LokiDataset.logger.error(f"Failed to load all required data for frame {frame_index}")
            return None

        LokiDataset.logger.info(f"Successfully loaded all data for frame {frame_index}")
        return (
            pcd,
            odometry,
            label3d
        )

    def load_loki_csv(self):
        loki_df = pd.read_csv(self.loki_csv_path)
        return loki_df
