import threading

import open3d as o3d

from modules.config.logger import Logger


class ThreadedPointCloudSaver:
    """
    Manages saving point clouds using threading for performance optimization.
    """

    def __init__(self):
        """
        Initializes the ThreadedPointCloudSaver by setting up a logger and a list for threads.
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.threads = []

    def _save_point_cloud(self, file_path: str, point_cloud: o3d.geometry.PointCloud) -> None:
        """
        Internal method to handle the actual saving of the point cloud.

        Args:
            file_path (str): The path where the point cloud will be saved.
            point_cloud (o3d.geometry.PointCloud): The point cloud to save.

        Raises:
            RuntimeError: If the point cloud could not be written to the specified file path.
        """
        try:
            success = o3d.io.write_point_cloud(file_path, point_cloud, write_ascii=False)
            if not success:
                raise RuntimeError(f"Failed to write point cloud to {file_path}")
            self.logger.info(f"Saved point cloud to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving file {file_path}: {e}")

    def add_save_task(self, file_path: str, point_cloud: o3d.geometry.PointCloud) -> None:
        """
        Creates and starts a new thread to save the specified point cloud to a file.

        Args:
            file_path (str): The path where the point cloud will be saved.
            point_cloud (o3d.geometry.PointCloud): The point cloud to save.
        """
        thread = threading.Thread(target=self._save_point_cloud, args=(file_path, point_cloud))
        self.threads.append(thread)
        thread.start()

    def wait_for_completion(self) -> None:
        """
        Blocks until all spawn threads have finished execution.
        """
        for thread in self.threads:
            thread.join()
        self.logger.info("All save tasks have completed.")
