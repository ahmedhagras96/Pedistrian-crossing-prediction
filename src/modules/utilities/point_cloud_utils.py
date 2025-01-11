import numpy as np
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler

from modules.utilities.base_utility import BaseUtility


class PointCloudUtils(BaseUtility):
    """
    A utility class for loading, normalizing, and preprocessing point cloud data.
    """

    def __init__(self):
        super().__init__()

    def load_ply_ndarray(self, file_path: str) -> np.ndarray:
        """
        Load a .ply file and return its point cloud data.

        Args:
            file_path (str): Path to the .ply file.

        Returns:
            np.ndarray: Point cloud data of shape (N, 3), where N is the number of points.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded file does not contain valid point cloud data.
        """
        self.logger.info(f"Loading point cloud from {file_path}")

        try:
            point_cloud = o3d.io.read_point_cloud(file_path)
            points = np.asarray(point_cloud.points)

            if points.size == 0:
                raise ValueError(f"No points found in the file: {file_path}")

            self.logger.info(f"Successfully loaded {points.shape[0]} points from {file_path}")
            return points

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            self.logger.error(f"Error loading point cloud from {file_path}: {e}")
            raise e

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud data to the range [0, 1] using Min-Max scaling.

        Args:
            points (np.ndarray): Input point cloud data of shape (N, 3).

        Returns:
            np.ndarray: Normalized point cloud data of shape (N, 3).

        Raises:
            ValueError: If the input points are not a valid 2D NumPy array.
        """
        self.logger.info("Normalizing point cloud data")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points must be a 2D NumPy array with shape (N, 3)")

        try:
            scaler = MinMaxScaler()
            normalized_points = scaler.fit_transform(points)
            self.logger.info("Normalization successful")
            return normalized_points

        except Exception as e:
            self.logger.error(f"Error normalizing points: {e}")
            raise e
