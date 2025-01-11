import open3d as o3d

from modules.base_module import BaseModule


class Open3DUtils(BaseModule):
    """
    A utility class for working with Open3D PointCloud objects.
    """

    def __init__(self):
        super().__init__()

    def convert_from_vertex_to_open3d_pcd(self, vertex_data):
        """
        Converts vertex data to an Open3D PointCloud object.

        Args:
            vertex_data (numpy.ndarray): A 2D array where each row represents a point.
                The first three columns (x, y, z) are mandatory.
                Optional additional columns (r, g, b) represent colors.

        Returns:
            o3d.geometry.PointCloud: The resulting Open3D PointCloud object.

        Raises:
            ValueError: If `vertex_data` is not a 2D array or does not have at least 3 columns.
        """
        self.logger.debug(f"Starting conversion with vertex_data of shape: {vertex_data.shape}")

        # Validate input data dimensions
        if vertex_data.ndim != 2 or vertex_data.shape[1] < 3:
            self.logger.error("vertex_data validation failed: must be a 2D array with at least 3 columns.")
            raise ValueError("vertex_data must be a 2D array with at least 3 columns (x, y, z coordinates).")

        # Extract XYZ coordinates
        xyz = vertex_data[:, :3]
        self.logger.debug("Extracted XYZ coordinates from vertex_data.")

        # Create PointCloud and set points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        self.logger.debug("Created Open3D PointCloud object with points.")

        # Set colors if available
        if vertex_data.shape[1] >= 6:
            colors = vertex_data[:, 3:6] / 255.0  # Normalize RGB values to [0, 1]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.logger.debug("Assigned colors to PointCloud object.")

        self.logger.info("Successfully converted vertex data to Open3D PointCloud.")
        return pcd
