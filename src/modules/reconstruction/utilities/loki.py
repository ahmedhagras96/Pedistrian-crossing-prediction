import os
import re
from typing import Tuple, Optional

import open3d as o3d
import pandas as pd

from modules.config.logger import Logger
from modules.reconstruction.utilities.file_utils import FileUtils
from modules.reconstruction.utilities.pointcloud_utils import PointCloudUtils


class LokiDataset:
    """
    Class to handle loading of LOKI dataset files including point clouds,
    odometry, and 3D label data.

    Attributes:
        scenario_path (str): Path to the specific scenario directory 
            (e.g., '/path/to/scenario_026').
        loki_csv_path (str): Path to the loki.csv file.
        scenario_id (Optional[str]): Numeric ID extracted from the scenario path, if available.
    """

    _logger = Logger.get_logger("LokiDataset")

    def __init__(self, scenario_path: Optional[str] = None, loki_csv_path: str = ""):
        """
        Initialize the LokiDataset with an optional scenario path and a LOKI CSV path.

        Args:
            scenario_path (str, optional): Full path to a scenario directory 
                (e.g., '/some/path/scenario_026'). If not provided, scenario_id remains None.
            loki_csv_path (str): Full path to the loki.csv file.

        Returns:
            None
        """
        self.scenario_path = scenario_path
        self.loki_csv_path = loki_csv_path
        self.scenario_id = None

        # Extract scenario ID only if a scenario path is provided
        if scenario_path:
            self.scenario_id = self._extract_scenario_id(scenario_path)

        if self.scenario_id:
            self._logger.info(f"Initialized {self.__class__.__name__} for scenario: {self.scenario_id}")
        else:
            self._logger.info(f"Initialized {self.__class__.__name__} without a valid scenario ID.")

    @staticmethod
    def _extract_scenario_id(path: str) -> Optional[str]:
        """
        Extract the numeric scenario ID from a path string.

        Args:
            path (str): A file system path, typically containing 'scenario_XXX'.

        Returns:
            Optional[str]: The extracted scenario ID as a string, or None if not found.
        """
        match = re.search(r'scenario_(\d+)', os.path.basename(path), re.IGNORECASE)
        return match.group(1) if match else None

    def load_ply(self, frame_index: int) -> Optional[o3d.geometry.PointCloud]:
        """
        Load the point cloud (.ply) file for the specified frame index.

        Args:
            frame_index (int): Frame index to load (e.g., 42 for 'pc_0042.ply').

        Returns:
            Optional[o3d.geometry.PointCloud]: The loaded point cloud, 
            or None if loading fails or the file does not exist.
        """
        ply_filename = f"pc_{frame_index:04d}.ply"
        ply_filepath = os.path.join(self.scenario_path, ply_filename)

        if not FileUtils.file_exists(ply_filepath):
            self._logger.error(f"Point cloud file does not exist: {ply_filepath}")
            return None

        try:
            return PointCloudUtils.load_point_cloud(ply_filepath)
        except Exception as exc:
            self._logger.error(f"Failed to load point cloud for frame {frame_index}: {exc}")
            return None

    def load_odometry(self, frame_index: int) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Load the odometry data for the specified frame index.

        Args:
            frame_index (int): Frame index to load (e.g., 42 for 'odom_0042.txt').

        Returns:
            Optional[Tuple[float, float, float, float, float, float]]: A 6-tuple of floats 
            (x, y, z, roll, pitch, yaw), or None if loading fails or the file does not exist.
        """
        odom_filename = f"odom_{frame_index:04d}.txt"
        odom_filepath = os.path.join(self.scenario_path, odom_filename)

        if not FileUtils.file_exists(odom_filepath):
            self._logger.error(f"Odometry file does not exist: {odom_filepath}")
            return None

        try:
            with open(odom_filepath, "r") as odom_file:
                data = odom_file.read().strip().split(",")
            if len(data) < 6:
                self._logger.error(f"Odometry file is malformed: {odom_filepath}")
                return None

            return tuple(map(float, data[:6]))
        except Exception as exc:
            self._logger.error(f"Failed to load odometry for frame {frame_index}: {exc}")
            return None

    def load_label3d(self, frame_index: int) -> Optional[pd.DataFrame]:
        """
        Load the label3D data for the specified frame index.

        Args:
            frame_index (int): Frame index to load (e.g., 42 for 'label3d_0042.txt').

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing label3D data, or None if loading fails.
            An empty DataFrame is returned if the file exists but is empty.
        """
        label3d_filename = f"label3d_{frame_index:04d}.txt"
        label3d_filepath = os.path.join(self.scenario_path, label3d_filename)

        if not FileUtils.file_exists(label3d_filepath):
            self._logger.error(f"Label3D file does not exist: {label3d_filepath}")
            return None

        try:
            with open(label3d_filepath, "r") as label_file:
                lines = label_file.readlines()
            if not lines:
                self._logger.warning(f"Label3D file is empty: {label3d_filepath}")
                return pd.DataFrame()

            # First line is assumed to be column headers
            columns = [col.strip() for col in lines[0].strip().split(",")]
            data = [line.strip().split(",") for line in lines[1:]]
            return pd.DataFrame(data, columns=columns)
        except Exception as exc:
            self._logger.error(f"Failed to load label3D for frame {frame_index}: {exc}")
            return None

    def load_alignment_data(
            self,
            frame_index: int,
            scenario_path: Optional[str] = None
    ) -> Optional[Tuple[
        o3d.geometry.PointCloud,
        Tuple[float, float, float, float, float, float],
        pd.DataFrame
    ]]:
        """
        Load the point cloud, odometry, and 3D label data for the specified frame index.

        Args:
            frame_index (int): Frame index to load.
            scenario_path (str, optional): Optional override path to a scenario directory. 
                If provided, updates self.scenario_path.

        Returns:
            Optional[Tuple[
                o3d.geometry.PointCloud,
                Tuple[float, float, float, float, float, float],
                pd.DataFrame
            ]]: A tuple containing (point_cloud, odometry, label3D dataframe),
            or None if any of those fails to load.
        """
        if scenario_path:
            self.scenario_path = scenario_path
            self.scenario_id = self._extract_scenario_id(scenario_path)
        elif not self.scenario_path:
            self._logger.error("No scenario path provided or set for the dataset.")
            return None

        self._logger.info(f"Loading all data for frame {frame_index} in scenario_id: {self.scenario_id}")

        pcd = self.load_ply(frame_index)
        odometry = self.load_odometry(frame_index)
        label3d = self.load_label3d(frame_index)

        if pcd is None or odometry is None or label3d is None:
            self._logger.error(f"Failed to load all required data for frame {frame_index}.")
            return None

        self._logger.info(f"Successfully loaded point cloud, odometry, and label3D for frame {frame_index}.")
        return pcd, odometry, label3d

    def load_loki_csv(self) -> pd.DataFrame:
        """
        Load the main LOKI CSV file.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the CSV data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        self._logger.info(f"Loading LOKI CSV from {self.loki_csv_path}")
        if not FileUtils.file_exists(self.loki_csv_path):
            raise FileNotFoundError(f"LOKI CSV file does not exist: {self.loki_csv_path}")

        return pd.read_csv(self.loki_csv_path)
