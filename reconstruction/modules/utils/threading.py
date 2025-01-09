import time
import threading
import os
import open3d as o3d
from .logger import Logger

class ThreadedSaveManager:
    """
    A class to manage saving point clouds using threading for performance optimization.
    """

    def __init__(self):
        """
        Initializes the ThreadedSaveManager.

        Args:
            logger: Logger instance to log information and errors.
        """
        # Initialize a class-level logger
        logger = Logger.get_logger(__name__)
        self.logger = logger
        self.threads = []

    def save_point_cloud(self, file_path, point_cloud):
        """
        Saves a point cloud to the specified file path.

        Args:
            file_path (str): Path to save the point cloud.
            point_cloud (o3d.geometry.PointCloud): The point cloud to save.
        """
        try:
            success = o3d.io.write_point_cloud(file_path, point_cloud, write_ascii=False)
            if not success:
                raise RuntimeError(f"Failed to write point cloud to {file_path}")
            self.logger.info(f"Saved point cloud to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving file {file_path}: {e}")

    def add_save_task(self, file_path, point_cloud):
        """
        Adds a save task to the thread pool.

        Args:
            file_path (str): Path to save the point cloud.
            point_cloud (o3d.geometry.PointCloud): The point cloud to save.
        """
        thread = threading.Thread(target=self.save_point_cloud, args=(file_path, point_cloud))
        self.threads.append(thread)
        thread.start()

    def wait_for_completion(self):
        """
        Waits for all threads to complete.
        """
        for thread in self.threads:
            thread.join()
        self.logger.info("All save tasks have completed.")
