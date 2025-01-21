from typing import List, Optional

import numpy as np
import open3d as o3d

from modules.config.logger import Logger
from modules.reconstruction.utilities.recon_3d_config import Reconstuction3DConfig


class PointCloudVisualizer:
    """
    A utility class for visualizing point clouds using Open3D with integrated custom logging.
    """

    def __init__(
            self,
            window_name: str = "Open3D Point Cloud Visualizer",
            point_size: Optional[float] = None,
            background_color: Optional[List[float]] = None
    ) -> None:
        """
        Initializes the PointCloudVisualizer with specified visualization parameters.
        
        Args:
            window_name (str, optional): Title of the visualization window.
                Defaults to "Open3D Point Cloud Visualizer".
            point_size (float, optional): Size of the points in the visualization.
                Defaults to value from Reconstuction3DConfig.
            background_color (List[float], optional): Background color of the visualization window
                as [R, G, B], each in [0.0, 1.0]. Defaults to value from Reconstuction3DConfig.
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.window_name = window_name

        if background_color is None:
            background_color = Reconstuction3DConfig.background_color
        if point_size is None:
            point_size = Reconstuction3DConfig.point_size

        self.background_color = background_color
        self.point_size = point_size
        self.vis = None  # Open3D Visualizer instance

        self.logger.info(
            f"Initialized {self.__class__.__name__} with window_name='{self.window_name}', "
            f"point_size={self.point_size}, background_color={self.background_color}"
        )

    def _initialize_visualizer(self) -> None:
        """
        Initializes the Open3D visualizer window with the specified settings.

        Raises:
            RuntimeError: If the Open3D visualizer cannot be created.
        """
        if self.vis is not None:
            self.logger.warning("Visualizer is already initialized.")
            return

        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.window_name)
            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size
            render_option.background_color = np.asarray(self.background_color)
            self.logger.info("Open3D visualizer window created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Open3D visualizer: {e}")
            self.vis = None
            raise RuntimeError(f"Failed to initialize Open3D visualizer: {e}")

    def add_point_cloud(
            self,
            pcd: o3d.geometry.PointCloud,
            color: Optional[List[float]] = None
    ) -> None:
        """
        Adds a point cloud to the visualizer with optional coloring.
        
        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to add.
            color (List[float], optional): [R, G, B] color to paint the point cloud.
                Each value must be in the range [0.0, 1.0]. If None, no painting is applied.
        
        Raises:
            ValueError: If the color list is not exactly three floats in the range [0, 1].
            RuntimeError: If the point cloud cannot be added to the visualizer.
        """
        if self.vis is None:
            self._initialize_visualizer()

        try:
            if color is not None:
                if len(color) != 3 or not all(0.0 <= c <= 1.0 for c in color):
                    self.logger.error("Color must be a list of three floats between 0.0 and 1.0.")
                    raise ValueError("Color must be a list of three floats between 0.0 and 1.0.")
                pcd.paint_uniform_color(color)
                self.logger.info(f"Painted point cloud with color: {color}")

            self.vis.add_geometry(pcd)
            self.logger.info("Added point cloud to the visualizer.")
        except Exception as e:
            self.logger.error(f"Failed to add point cloud to visualizer: {e}")
            raise RuntimeError(f"Failed to add point cloud to visualizer: {e}")

    def run(self) -> None:
        """
        Runs the Open3D visualizer window. This call is blocking until the window is closed.

        Raises:
            RuntimeError: If there is an error during visualization execution.
        """
        if self.vis is None:
            self._initialize_visualizer()

        try:
            self.logger.info("Starting the visualization window. Close the window to continue.")
            self.vis.run()
            self.logger.info("Visualization window closed.")
        except Exception as e:
            self.logger.error(f"Error during visualization run: {e}")
            raise RuntimeError(f"Error during visualization run: {e}")

    def close(self) -> None:
        """
        Closes the Open3D visualizer window if it's open.

        Raises:
            RuntimeError: If there is an error destroying the window.
        """
        if self.vis is not None:
            try:
                self.vis.destroy_window()
                self.logger.info("Open3D visualizer window destroyed successfully.")
            except Exception as e:
                self.logger.error(f"Failed to destroy Open3D visualizer window: {e}")
                raise RuntimeError(f"Failed to destroy Open3D visualizer window: {e}")
            finally:
                self.vis = None
        else:
            self.logger.warning("Visualizer is not initialized or already closed.")

    def visualize(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Convenience method to visualize a single point cloud and then close the window.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to visualize.
        """
        self.add_point_cloud(pcd)
        self.run()
        self.close()
