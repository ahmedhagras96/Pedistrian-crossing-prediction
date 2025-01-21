from typing import Tuple

import numpy as np
import open3d as o3d
import pandas as pd

from modules.config.logger import Logger
from modules.reconstruction.aligners.base_aligner import BaseAligner
from modules.reconstruction.aligners.align_direction import AlignDirection
from modules.reconstruction.aligners.odometry_validation import OdometryValidation
from modules.reconstruction.utilities.pointcloud_utils import PointCloudUtils
from modules.reconstruction.utilities.recon_3d_config import Reconstuction3DConfig


class PointCloudOdometryAligner(BaseAligner):
    """
    Aligner for point clouds based on vehicle odometry data.

    Uses odometry information to spatially align both environment and object
    point clouds for improved accuracy in 3D reconstructions.
    """

    def __init__(self, scenario_path: str, loki_csv_path: str):
        """
        Initialize the PointCloudOdometryAligner with scenario and odometry data.

        Args:
            scenario_path (str): Path to the scenario directory containing point cloud and odometry files.
            loki_csv_path (str): Path to the LOKI CSV file containing additional data.
        """
        super().__init__(scenario_path, loki_csv_path)
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def align(
            self,
            key_frame: int,
            align_interval: int = 10,
            align_direction: AlignDirection = AlignDirection.SPLIT
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Align point clouds based on the specified key frame, interval, and direction.

        Args:
            key_frame (int): The reference frame index for alignment.
            align_interval (int, optional): Number of frames to align on either side of the key frame. Defaults to 10.
            align_direction (AlignDirection, optional): Direction of alignment (LEFT, RIGHT, SPLIT). 
                                                       Defaults to AlignDirection.SPLIT.

        Returns:
            Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]: 
                (aligned environment point cloud, aligned objects point cloud).

        Raises:
            ValueError: If alignment parameters are invalid based on odometry validation.
        """
        self.logger.info("Starting Odometry Alignment Process")

        # Validate alignment inputs
        OdometryValidation.validate_alignment_input(
            key_frame, align_interval, align_direction, self.max_frames
        )
        self.logger.info("All input parameters validated successfully.")
        self.logger.info(f"Key Frame: {key_frame}, Interval: {align_interval}, Direction: {align_direction.name}")

        # Determine start and end frames
        start_frame, end_frame = self._get_start_end_frames(key_frame, align_interval, align_direction)
        self.logger.info(f"Selected frame range: {start_frame} to {end_frame} (step={self.frame_step}).")

        aligned_env = self._align_environment(start_frame, end_frame)
        aligned_objects = self._align_last_objects_instance(end_frame)

        return aligned_env, aligned_objects

    def _align_environment(self, start_frame: int, end_frame: int) -> o3d.geometry.PointCloud:
        """
        Align the environment point clouds between the specified start and end frames.

        Loads, optionally downsamples, and applies odometry transformations to each frame's
        point cloud to produce an aggregated, aligned environment.

        Args:
            start_frame (int): The starting frame index for environment alignment.
            end_frame (int): The ending frame index for environment alignment.

        Returns:
            o3d.geometry.PointCloud: Aggregated aligned environment point cloud.
        """
        self.logger.info("Starting Odometry Environment Alignment Process")
        frame_indices = list(map(int, np.arange(start_frame, end_frame + 1, self.frame_step)))
        self.logger.info(f"Processing frames: {frame_indices}")

        environment_pcd_queue = []
        for frame_index in frame_indices:
            self.logger.info(f"Loading frame {frame_index}")
            pcd = self.loki.load_ply(frame_index)
            odom = self.loki.load_odometry(frame_index)
            label3d = self.loki.load_label3d(frame_index)

            if Reconstuction3DConfig.use_downsampling:
                pcd = pcd.voxel_down_sample(voxel_size=Reconstuction3DConfig.voxel_size)

            # Remove objects from the environment
            pcd = self._remove_objects_from_environment(pcd, label3d)

            # Apply odometry transformation
            transformation_matrix = self.get_transformation_matrix(odom)
            pcd.transform(transformation_matrix)

            environment_pcd_queue.append(pcd)

        # Aggregate all aligned environment point clouds
        aligned_environment = o3d.geometry.PointCloud()
        for pcd in environment_pcd_queue:
            aligned_environment += pcd

        self.logger.info("Successfully aligned environment point clouds using odometry.")
        return aligned_environment

    def _align_last_objects_instance(self, end_frame: int) -> o3d.geometry.PointCloud:
        """
        Align and extract objects (tracked entities) from the specified end frame.

        Loads odometry, label3D data, and the point cloud for the end frame, applies
        transformations, and aggregates all relevant objects.

        Args:
            end_frame (int): The frame index from which to align objects.

        Returns:
            o3d.geometry.PointCloud: Aggregated aligned objects point cloud.
        """
        self.logger.info("Starting Last Objects Instances Alignment Process")
        self.logger.info(f"Extracting and aligning objects from frame {end_frame}")

        pcd, odom, label3d = self.loki.load_alignment_data(end_frame)
        label3d = label3d[label3d['labels'].isin(Reconstuction3DConfig.tracked_objects)]

        positions = label3d[['pos_x', 'pos_y', 'pos_z']].values
        dimensions = label3d[['dim_x', 'dim_y', 'dim_z']].values
        yaws = label3d['yaw'].values

        # Crop each tracked object
        objects_pcd = o3d.geometry.PointCloud()
        for pos, dims, yaw in zip(positions, dimensions, yaws):
            cropped_obj = PointCloudUtils.crop_pcd(pcd, pos, dims, yaw)
            if cropped_obj:
                objects_pcd += cropped_obj

        # Transform the combined objects using odometry
        transformation_matrix = self.get_transformation_matrix(odom)
        objects_pcd.transform(transformation_matrix)

        self.logger.info("Successfully aligned objects from the last frame using odometry.")
        return objects_pcd

    def _remove_objects_from_environment(self, pcd: o3d.geometry.PointCloud,
                                         label3d_df: pd.DataFrame) -> o3d.geometry.PointCloud:
        """
        Remove specified objects from the environment point cloud based on label3D data.

        Args:
            pcd (o3d.geometry.PointCloud): The original environment point cloud.
            label3d_df (pd.DataFrame): DataFrame containing label3D information for objects.

        Returns:
            o3d.geometry.PointCloud: Environment point cloud with specified objects removed.
        """
        positions = label3d_df[['pos_x', 'pos_y', 'pos_z']].values
        dimensions = label3d_df[['dim_x', 'dim_y', 'dim_z']].values
        yaws = label3d_df['yaw'].values

        count_removed = 0
        for pos, dims, yaw in zip(positions, dimensions, yaws):
            pcd = PointCloudUtils.crop_pcd(pcd, pos, dims, yaw, remove=True)
            count_removed += 1

        self.logger.debug(f"Removed {count_removed} objects from the environment.")
        return pcd

    def _get_start_end_frames(
            self, key_frame: int, align_interval: int, align_direction: AlignDirection
    ) -> Tuple[int, int]:
        """
        Determine start/end frames based on key frame, interval, and alignment direction.

        Args:
            key_frame (int): The reference frame for alignment.
            align_interval (int): Number of frames on each side of the key frame.
            align_direction (AlignDirection): Direction of alignment (LEFT, RIGHT, SPLIT).

        Returns:
            Tuple[int, int]: Calculated start and end frame indices.
        """
        align_interval *= 2  # double alignment to account for even-only frames

        if align_direction == AlignDirection.LEFT:
            start_frame = key_frame - align_interval
            end_frame = key_frame
        elif align_direction == AlignDirection.RIGHT:
            start_frame = key_frame
            end_frame = key_frame + align_interval
        elif align_direction == AlignDirection.SPLIT:
            half_interval = align_interval // 2
            start_frame = key_frame - half_interval
            end_frame = key_frame + half_interval
        else:
            self.logger.error(f"Invalid alignment direction: {align_direction}")
            raise ValueError("Invalid alignment direction specified.")

        # Adjust start_frame if odd
        if start_frame % 2 != 0:
            self.logger.warning(f"Start frame {start_frame} is odd. Adjusting to the next even frame.")
            start_frame += 1

        # Adjust end_frame if odd
        if end_frame % 2 != 0:
            self.logger.warning(f"End frame {end_frame} is odd. Adjusting to the previous even frame.")
            end_frame -= 1

        return start_frame, end_frame
