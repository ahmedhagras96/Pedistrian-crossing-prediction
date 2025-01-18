from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler

from modules.config.logger import LoggerUtils


class PointCloudUtils:
    """
    A utility class for loading, normalizing, and preprocessing point cloud data.
    All methods are static and use a class-level logger.
    """
    # Initialize a class-level logger.
    logger = LoggerUtils.get_logger("PointCloudUtils")

    @staticmethod
    def load_ply_ndarray(file_path: str) -> np.ndarray:
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
        logger = PointCloudUtils.logger
        logger.info(f"Loading point cloud from {file_path}")

        try:
            point_cloud = o3d.io.read_point_cloud(file_path)
            points = np.asarray(point_cloud.points)

            if points.size == 0:
                raise ValueError(f"No points found in the file: {file_path}")

            logger.info(f"Successfully loaded {points.shape[0]} points from {file_path}")
            return points

        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise e
        except Exception as e:
            logger.error(f"Error loading point cloud from {file_path}: {e}")
            raise e

    @staticmethod
    def normalize_points(points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud data to the range [0, 1] using Min-Max scaling.

        Args:
            points (np.ndarray): Input point cloud data of shape (N, 3).

        Returns:
            np.ndarray: Normalized point cloud data of shape (N, 3).

        Raises:
            ValueError: If the input points are not a valid 2D NumPy array.
        """
        logger = PointCloudUtils.logger
        logger.info("Normalizing point cloud data")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points must be a 2D NumPy array with shape (N, 3)")

        try:
            scaler = MinMaxScaler()
            normalized_points = scaler.fit_transform(points)
            logger.info("Normalization successful")
            return normalized_points

        except Exception as e:
            logger.error(f"Error normalizing points: {e}")
            raise e

    @staticmethod
    def load_and_normalize_ply(ply_file: Path) -> np.ndarray:
        """
        Load and normalize 3D points from a .ply file.
        
        Args:
            ply_file (str): Path to the .ply file.
        
        Returns:
            np.ndarray: Normalized 3D points of shape (N, 3).
        """
        logger = PointCloudUtils.logger
        logger.info(f"Loading and normalizing point cloud from {ply_file}")

        try:
            # Load point cloud from file.
            ply_file = str(ply_file)
            point_cloud = o3d.io.read_point_cloud(ply_file)
            points = np.asarray(point_cloud.points)

            # Check if points are valid.
            if points.size == 0:
                raise ValueError(f"No points found in the file: {ply_file}")

            # Normalize the points.
            scaler = MinMaxScaler()
            normalized_points = scaler.fit_transform(points)
            logger.info(f"Loaded and normalized {points.shape[0]} points from {ply_file}")

            return normalized_points

        except Exception as e:
            logger.error(f"Error processing {ply_file}: {e}")
            raise e
