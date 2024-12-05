# utils/point_cloud_manager.py
import open3d as o3d
import os
from typing import Optional, List

from .logger import Logger

class PointCloudUtils:
    """
    Utility class for handling point cloud operations using Open3D.
    """

    # Initialize a class-level logger
    logger = Logger.get_logger(__name__)

    @staticmethod
    def load_point_cloud(file_path: str) -> Optional[o3d.geometry.PointCloud]:
        """
        Load a point cloud from a file.

        Args:
            file_path (str): Path to the point cloud file.

        Returns:
            Optional[o3d.geometry.PointCloud]: Loaded point cloud or None if failed.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the point cloud cannot be loaded.
        """
        if not os.path.isfile(file_path):
            PointCloudUtils.logger.error(f"Point cloud file {file_path} does not exist.")
            raise FileNotFoundError(f"Point cloud file {file_path} does not exist.")

        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                PointCloudUtils.logger.error(f"Point cloud {file_path} contains no points.")
                raise ValueError(f"Point cloud {file_path} contains no points.")
            PointCloudUtils.logger.info(f"Loaded point cloud from {file_path}")
            return pcd
        except Exception as e:
            PointCloudUtils.logger.error(f"Failed to load point cloud from {file_path}: {e}")
            raise RuntimeError(f"Failed to load point cloud from {file_path}: {e}")

    @staticmethod
    def save_point_cloud(point_cloud: o3d.geometry.PointCloud, file_path: str) -> None:
        """
        Save a point cloud to a file.

        Args:
            point_cloud (o3d.geometry.PointCloud): The point cloud to save.
            file_path (str): Path where the point cloud will be saved.

        Raises:
            RuntimeError: If the point cloud cannot be saved.
        """
        try:
            success = o3d.io.write_point_cloud(file_path, point_cloud)
            if not success:
                PointCloudUtils.logger.error(f"Open3D failed to write point cloud to {file_path}.")
                raise RuntimeError(f"Open3D failed to write point cloud to {file_path}.")
            PointCloudUtils.logger.info(f"Saved point cloud to {file_path}")
        except Exception as e:
            PointCloudUtils.logger.error(f"Failed to save point cloud to {file_path}: {e}")
            raise RuntimeError(f"Failed to save point cloud to {file_path}: {e}")

    @staticmethod
    def visualize_point_clouds(point_clouds: List[o3d.geometry.PointCloud], window_name: str = "Point Cloud Visualization"):
        """
        Visualize multiple point clouds.

        Args:
            point_clouds (List[o3d.geometry.PointCloud]): List of point clouds to visualize.
            window_name (str, optional): Name of the visualization window. Defaults to "Point Cloud Visualization".
        """
        if not point_clouds:
            PointCloudUtils.logger.warning("No point clouds provided for visualization.")
            return

        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name)
            PointCloudUtils.logger.info("Created Open3D visualization window")

            for pcd in point_clouds:
                vis.add_geometry(pcd)
                PointCloudUtils.logger.debug("Added point cloud to visualization")

            PointCloudUtils.logger.info("Starting visualization...")
            vis.run()
            PointCloudUtils.logger.info("Visualization finished")
            vis.destroy_window()
            PointCloudUtils.logger.debug("Destroyed Open3D visualization window")
        except Exception as e:
            PointCloudUtils.logger.error(f"Visualization failed: {e}")
            raise RuntimeError(f"Visualization failed: {e}")
