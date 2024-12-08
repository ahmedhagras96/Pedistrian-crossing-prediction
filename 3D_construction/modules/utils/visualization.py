import open3d as o3d
import numpy as np
from typing import List

from .logger import Logger
from .recon_3d_config import Reconstuction3DConfig

class PointCloudVisualizer:
    """
    A utility class for visualizing point clouds using Open3D with integrated custom logging.
    """

    # Initialize a class-level logger
    logger = Logger.get_logger(__name__)

    def __init__(self, window_name: str = "Open3D Point Cloud Visualizer",
                 point_size: float = None,
                 background_color=None):
        """
        Initializes the PointCloudVisualizer with specified visualization parameters.
        
        Args:
            window_name (str, optional): Title of the visualization window. Defaults to "Open3D Point Cloud Visualizer".
            point_size (float, optional): Size of the points in the visualization. Defaults to 3.0.
            background_color (List[float], optional): Background color of the visualization window. Defaults to [0.2, 0.2, 0.2].
        """
        self.window_name = window_name
        
        if background_color is None:
            background_color = Reconstuction3DConfig.background_color
            
        if point_size is None:
            point_size = Reconstuction3DConfig.point_size
        
        self.background_color = background_color
        self.point_size = point_size
        self.vis = None  # Open3D Visualizer instance
        # PointCloudVisualizer.logger.info(f"Initialized PointCloudVisualizer with window_name='{self.window_name}', "
        #                  f"point_size={self.point_size}, background_color={self.background_color}")

    def _initialize_visualizer(self):
        """
        Initializes the Open3D visualizer window with the specified settings.
        """
        if self.vis is not None:
            # PointCloudVisualizer.logger.warning("Visualizer is already initialized.")
            return

        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.window_name)
            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size
            render_option.background_color = np.asarray(self.background_color)
            # PointCloudVisualizer.logger.info("Open3D visualizer window created successfully.")
        except Exception as e:
            # PointCloudVisualizer.logger.error(f"Failed to initialize Open3D visualizer: {e}")
            self.vis = None
            raise RuntimeError(f"Failed to initialize Open3D visualizer: {e}")

    def visualize(self, pcd: o3d.geometry.PointCloud):
        self.add_point_cloud(pcd)
        self.run()
        self.close()
    
    def add_point_cloud(self, pcd: o3d.geometry.PointCloud, color: List[float] = None):
        """
        Adds a point cloud to the visualizer with optional coloring.
        
        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to add.
            color (Optional[List[float]], optional): RGB color to paint the point cloud. Defaults to None.
        """
        if self.vis is None:
            self._initialize_visualizer()

        try:
            if color:
                if len(color) != 3 or not all(0.0 <= c <= 1.0 for c in color):
                    # PointCloudVisualizer.logger.error("Color must be a list of three floats between 0 and 1.")
                    raise ValueError("Color must be a list of three floats between 0 and 1.")
                pcd.paint_uniform_color(color)
                # PointCloudVisualizer.logger.info(f"Painted point cloud with color: {color}")
            # else:
                # Assign a default color if none is provided
                # pcd.paint_uniform_color([1, 1, 1])  # White
                # PointCloudVisualizer.logger.info("No color provided. Painted point cloud white by default.")

            self.vis.add_geometry(pcd)
            # PointCloudVisualizer.logger.info(f"Added point cloud to visualizer: {name if name else 'Unnamed'}")
        except Exception as e:
            # PointCloudVisualizer.logger.error(f"Failed to add point cloud to visualizer: {e}")
            raise RuntimeError(f"Failed to add point cloud to visualizer: {e}")

    def run(self):
        """
        Runs the Open3D visualizer window. This call is blocking until the window is closed.
        """
        if self.vis is None:
            self._initialize_visualizer()

        try:
            # PointCloudVisualizer.logger.info("Starting the visualization window. Close the window to continue.")
            self.vis.run()
            # PointCloudVisualizer.logger.info("Visualization window closed.")
        except Exception as e:
            # PointCloudVisualizer.logger.error(f"Error during visualization run: {e}")
            raise RuntimeError(f"Error during visualization run: {e}")

    def close(self):
        """
        Closes the Open3D visualizer window if it's open.
        """
        if self.vis is not None:
            try:
                self.vis.destroy_window()
                # PointCloudVisualizer.logger.info("Open3D visualizer window destroyed successfully.")
            except Exception as e:
                # PointCloudVisualizer.logger.error(f"Failed to destroy Open3D visualizer window: {e}")
                raise RuntimeError(f"Failed to destroy Open3D visualizer window: {e}")
            finally:
                self.vis = None
        else:
            # PointCloudVisualizer.logger.warning("Visualizer is not initialized or already closed.")
            pass
