import numpy as np
import open3d as o3d
import pandas as pd

from modules.config.logger import Logger


class BoundingBox:
    """
    Represents a bounding box in 3D space for a given row of data.
    """

    def __init__(self, row: pd.Series) -> None:
        """
        Initializes a BoundingBox object from a DataFrame row.

        Args:
            row (pd.Series): A row from a DataFrame containing bounding box information.
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.row = row
        self.obb = self._create_oriented_bounding_box()
        self.logger.info(f"Initialized {self.__class__.__name__} for row index {row.name}")

    def _create_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox or None:
        """
        Creates an Oriented Bounding Box (OBB) if all required fields are valid.

        Returns:
            o3d.geometry.OrientedBoundingBox or None:
                The oriented bounding box or None if required data is invalid.
        """
        required_fields = ['pos_x', 'pos_y', 'pos_z', 'dim_x', 'dim_y', 'dim_z', 'yaw']

        if not self._has_all_required_fields(required_fields):
            return None

        center = self._compute_center()
        extent = self._compute_extent()
        rotation_matrix = self._compute_rotation_matrix_z(self.row['yaw'])

        obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
        self.logger.info(f"Created Oriented Bounding Box for row index {self.row.name}")
        return obb

    def _has_all_required_fields(self, required_fields: list) -> bool:
        """
        Checks if all required fields are present and non-null in the row.

        Args:
            required_fields (list): List of field names to check in the row.

        Returns:
            bool: True if all required fields are valid, False otherwise.
        """
        for field in required_fields:
            if pd.isnull(self.row.get(field, None)):
                self.logger.warning(
                    f"Missing or null field '{field}' in row {self.row.name}. "
                    f"Skipping bounding box creation."
                )
                return False
        return True

    def _compute_center(self) -> np.ndarray:
        """
        Computes the bounding box center from the row data.

        Returns:
            np.ndarray: Center of the bounding box in [x, y, z] format.
        """
        center = np.array([self.row['pos_x'], self.row['pos_y'], self.row['pos_z']])
        return center

    def _compute_extent(self) -> np.ndarray:
        """
        Computes the bounding box extent (dimensions) from the row data.

        Returns:
            np.ndarray: Bounding box extent in [x, y, z] format.
        """
        extent = np.array([self.row['dim_x'], self.row['dim_y'], self.row['dim_z']])
        return extent

    def _compute_rotation_matrix_z(self, yaw: float) -> np.ndarray:
        """
        Computes the rotation matrix around the Z-axis using the yaw angle.

        Args:
            yaw (float): Rotation (in radians) around the Z-axis.

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        return rotation_matrix
