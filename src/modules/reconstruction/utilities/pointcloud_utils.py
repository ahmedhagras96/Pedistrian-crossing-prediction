import os
from typing import Optional, List, Tuple

import numpy as np
import open3d as o3d

from modules.config.logger import Logger


class PointCloudUtils:
    """
    Utility class for handling point cloud operations using Open3D.
    """

    _logger = Logger.get_logger("PointCloudUtils")

    @staticmethod
    def load_point_cloud(file_path: str) -> Optional[o3d.geometry.PointCloud]:
        """
        Load a point cloud from a file.

        Args:
            file_path (str): Path to the point cloud file.

        Returns:
            Optional[o3d.geometry.PointCloud]: The loaded point cloud, or None if loading fails.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file exists but contains no points.
            RuntimeError: If any other error occurs while loading.
        """
        if not os.path.isfile(file_path):
            PointCloudUtils._logger.error(f"Point cloud file '{file_path}' does not exist.")
            raise FileNotFoundError(f"Point cloud file '{file_path}' does not exist.")

        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                PointCloudUtils._logger.error(f"Point cloud '{file_path}' contains no points.")
                raise ValueError(f"Point cloud '{file_path}' contains no points.")
            return pcd
        except Exception as exc:
            PointCloudUtils._logger.error(f"Failed to load point cloud from '{file_path}': {exc}")
            raise RuntimeError(f"Failed to load point cloud from '{file_path}': {exc}")

    @staticmethod
    def save_point_cloud(point_cloud: o3d.geometry.PointCloud, file_path: str) -> None:
        """
        Save a point cloud to a file.

        Args:
            point_cloud (o3d.geometry.PointCloud): The point cloud to save.
            file_path (str): Path where the point cloud will be saved.

        Returns:
            None

        Raises:
            RuntimeError: If the point cloud cannot be saved.
        """
        try:
            success = o3d.io.write_point_cloud(file_path, point_cloud)
            if not success:
                PointCloudUtils._logger.error(f"Open3D failed to write point cloud to '{file_path}'.")
                raise RuntimeError(f"Open3D failed to write point cloud to '{file_path}'.")
            PointCloudUtils._logger.info(f"Saved point cloud to '{file_path}'.")
        except Exception as exc:
            PointCloudUtils._logger.error(f"Failed to save point cloud to '{file_path}': {exc}")
            raise RuntimeError(f"Failed to save point cloud to '{file_path}': {exc}")

    @staticmethod
    def visualize_point_clouds(
            point_clouds: List[o3d.geometry.PointCloud],
            window_name: str = "Point Cloud Visualization"
    ) -> None:
        """
        Visualize multiple point clouds using Open3D's Visualizer.

        Args:
            point_clouds (List[o3d.geometry.PointCloud]): A list of point clouds to visualize.
            window_name (str): Name of the visualization window. Defaults to "Point Cloud Visualization".

        Returns:
            None

        Raises:
            RuntimeError: If visualization fails due to an unexpected error.
        """
        if not point_clouds:
            PointCloudUtils._logger.warning("No point clouds provided for visualization.")
            return

        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name)
            PointCloudUtils._logger.info("Created Open3D visualization window.")

            for pcd in point_clouds:
                vis.add_geometry(pcd)

            PointCloudUtils._logger.info("Starting visualization...")
            vis.run()
            PointCloudUtils._logger.info("Visualization finished.")
            vis.destroy_window()
        except Exception as exc:
            PointCloudUtils._logger.error(f"Visualization failed: {exc}")
            raise RuntimeError(f"Visualization failed: {exc}")

    @staticmethod
    def crop_pcd(
            pcd: o3d.geometry.PointCloud,
            pos: Tuple[float, float, float],
            dim: Tuple[float, float, float],
            yaw: float,
            remove: bool = False
    ) -> o3d.geometry.PointCloud:
        """
        Crop (or optionally invert-crop) a point cloud using an oriented bounding box.

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud to crop.
            pos (Tuple[float, float, float]): The center position of the bounding box (x, y, z).
            dim (Tuple[float, float, float]): The dimensions of the bounding box (extent in x, y, z).
            yaw (float): Yaw angle (in radians) around the z-axis.
            remove (bool): If True, perform an inverse crop (i.e., keep points outside the box).
                Defaults to False.

        Returns:
            o3d.geometry.PointCloud: A new point cloud containing the cropped (or inverse-cropped) points.
        """
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw])
        bounding_box = o3d.geometry.OrientedBoundingBox(center=pos, R=rotation_matrix, extent=dim)
        cropped_pcd = pcd.crop(bounding_box, invert=remove)
        return cropped_pcd

    @staticmethod
    def get_transformation_matrix(odometry: Tuple[float, float, float, float, float, float]) -> np.ndarray:
        """
        Construct a 4x4 transformation matrix from odometry data.

        Args:
            odometry (Tuple[float, float, float, float, float, float]): 
                A tuple of 6 floats (x, y, z, roll, pitch, yaw).

        Returns:
            np.ndarray: A 4x4 NumPy array representing the transformation matrix.
        """
        x, y, z, roll, pitch, yaw = odometry
        translation = np.array([x, y, z])
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((roll, pitch, yaw))

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation

        return transformation_matrix

    @staticmethod
    def get_yaw_matrix(yaw: float) -> np.ndarray:
        """
        Create a 3x3 rotation matrix that rotates points around the z-axis by the given yaw angle.

        Args:
            yaw (float): Yaw angle in radians.

        Returns:
            np.ndarray: A 3x3 NumPy array representing the rotation around the z-axis.
        """
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        yaw_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ], dtype=float)

        return yaw_matrix
