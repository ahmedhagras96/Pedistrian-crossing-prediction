import sys

import open3d as o3d

from modules.avatar.utilities.pointcloud_utils import convert_from_vertex_to_open3d_pcd
from modules.config.logger import Logger


class PointCloudProcessor:
    """
    Handles preprocessing of point clouds, including conversion, downsampling, and outlier removal.
    """

    def __init__(self, voxel_size: float = 0.02, nb_neighbors: int = 20, std_ratio: float = 2.0) -> None:
        """
        Initializes the PointCloudProcessor with specified parameters.

        Args:
            voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.02.
            nb_neighbors (int, optional): Number of neighbors for statistical outlier removal. Defaults to 20.
            std_ratio (float, optional): Standard deviation ratio for statistical outlier removal. Defaults to 2.0.
        """
        # Replace 'Logger' with your actual logger class import or definition
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

        self.logger.info(f"Initialized {self.__class__.__name__} with "
                         f"voxel_size={voxel_size}, nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")

    def preprocess_pcd(self, raw_pcd) -> o3d.geometry.PointCloud:
        """
        Preprocesses the raw point cloud by converting, downsampling, and removing outliers.

        Args:
            raw_pcd: The raw point cloud data (format depends on the data source).

        Returns:
            o3d.geometry.PointCloud: The cleaned and downsampled point cloud.

        Raises:
            SystemExit: If the point cloud cannot be converted.
        """
        pcd = self._convert_raw_pcd_to_o3d(raw_pcd)
        pcd_downsampled = self._downsample_pcd(pcd)
        pcd_clean = self._remove_outliers(pcd_downsampled)
        return pcd_clean

    def _convert_raw_pcd_to_o3d(self, raw_pcd) -> o3d.geometry.PointCloud:
        """
        Converts the raw point cloud data to an Open3D PointCloud object.

        Args:
            raw_pcd: The raw point cloud data (format depends on the data source).

        Returns:
            o3d.geometry.PointCloud: The Open3D point cloud object.

        Raises:
            SystemExit: If the point cloud conversion fails.
        """
        try:
            pcd = convert_from_vertex_to_open3d_pcd(raw_pcd)
            self.logger.info("Successfully converted raw point cloud to Open3D format.")
            return pcd
        except ValueError as ve:
            self.logger.error(f"Error converting point cloud: {ve}")
            raise ve

    def _downsample_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Downsamples the point cloud using voxel downsampling.

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud.

        Returns:
            o3d.geometry.PointCloud: The downsampled point cloud.
        """
        self.logger.info("Downsampling the point cloud...")
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        self.logger.info("Downsampling completed.")
        return pcd_down

    def _remove_outliers(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Removes statistical outliers from the point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to be cleaned.

        Returns:
            o3d.geometry.PointCloud: The cleaned point cloud.
        """
        self.logger.info("Removing statistical outliers from the point cloud...")
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )
        pcd_clean = pcd.select_by_index(ind)
        self.logger.info("Outlier removal completed.")
        return pcd_clean
