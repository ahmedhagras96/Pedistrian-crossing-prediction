import os

import numpy as np
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler

from modules.config.logger import Logger


class PlyProcessor:
    """
    A utility class to handle loading and normalization of .ply files.
    """

    _logger = Logger.get_logger("PlyProcessor")

    @staticmethod
    def load_and_normalize_ply(ply_file: str) -> np.ndarray:
        """
        Load and normalize 3D points from a .ply file.

        Args:
            ply_file (str): Path to the .ply file.

        Returns:
            np.ndarray: Normalized 3D points of shape (N, 3).

        Raises:
            FileNotFoundError: If the specified ply_file does not exist.
            ValueError: If the point cloud could not be read or is empty.
        """
        PlyProcessor._logger.debug(f"Loading .ply file from: {ply_file}")

        if not os.path.exists(ply_file):
            raise FileNotFoundError(f"The file {ply_file} does not exist.")

        # Load the .ply file using Open3D
        point_cloud = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(point_cloud.points)  # Shape: (N, 3)

        if points.size == 0:
            raise ValueError(f"No points found in {ply_file}.")

        # PlyProcessor._logger.debug(f"Points loaded: {points.shape[0]}")

        # Normalize points to [0, 1] using Min-Max scaling
        scaler = MinMaxScaler()
        normalized_points = scaler.fit_transform(points)

        # PlyProcessor._logger.debug(f"Points normalized to [0, 1] range.")
        return normalized_points
