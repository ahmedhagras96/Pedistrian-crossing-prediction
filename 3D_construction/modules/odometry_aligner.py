from typing import Tuple
import numpy as np
import open3d as o3d
import pandas as pd

from .base_aligner import BaseAligner
from .utils.logger import Logger
from .utils.pointcloud_utils import PointCloudUtils
from .helpers.align_direction import AlignDirection
from .helpers.odometry_validation import OdometryValidation

class PointCloudOdometryAligner(BaseAligner):
    """
    Processor for aligning point clouds based on vehicle odometry.
    
    This class handles the alignment of environment and object point clouds 
    using odometry data to ensure accurate spatial relationships.
    """

    def __init__(self, scenario_path: str, loki_csv_path: str, key_frame: int, max_frames: int = 100, frame_step: int = 2):
        """
        Initializes the PointCloudOdometryAligner with scenario and odometry data.
        
        Args:
            scenario_path (str): Path to the scenario directory containing point cloud and odometry files.
            loki_csv_path (str): Path to the LOKI CSV file containing additional data.
            key_frame (int): The reference frame index for alignment.
            max_frames (int, optional): Maximum number of frames available for processing. Defaults to 100.
            frame_step (int, optional): Step size between frames to be aligned. Defaults to 2.
        
        Raises:
            ValueError: If key_frame is negative.
        """
        super().__init__(scenario_path, loki_csv_path, key_frame, max_frames, frame_step)

        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def align(self, key_frame: int = None, align_interval: int = 10, align_direction: AlignDirection = AlignDirection.SPLIT):
        """
        Aligns point clouds based on the specified key frame, alignment interval, and direction.
        
        This method validates input parameters, determines the frames to align, and 
        processes both environment and object point clouds.
        
        Args:
            key_frame (int, optional): The reference frame index for alignment. 
                                       If None, uses the instance's key_frame. Defaults to None.
            align_interval (int, optional): Number of frames to align on either side of the key frame. 
                                            Defaults to 10.
            align_direction (AlignDirection, optional): Direction of alignment (LEFT, RIGHT, SPLIT). 
                                                       Defaults to AlignDirection.SPLIT.
        
        Returns:
            Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]: 
                - Aligned environment point cloud.
                - Aligned objects point cloud.
        
        Raises:
            ValueError: If alignment parameters are invalid.
        """
        self.logger.info("Starting Odometry Alignment Process")

        key_frame = key_frame if key_frame is not None else self.key_frame

        OdometryValidation.validate_alignment_input(key_frame, align_interval, align_direction, self.max_frames, self.logger)
        self.logger.info("All input data valid")
        self.logger.info(f"Key frame set to frame {key_frame}")
        self.logger.info(f"Alignment Interval set to {align_interval}")
        self.logger.info(f"Alignment Direction set to {align_direction.name}")

        self.logger.info(f"Aligning {align_interval} frames in {align_direction.name} direction")
        start_frame, end_frame = self._get_start_end_frames(key_frame, align_interval, align_direction)

        aligned_pcd = o3d.geometry.PointCloud()
        aligned_env = self._align_environment(start_frame, end_frame)
        aligned_objects = self._align_last_objects_instance(end_frame)

        if aligned_env and aligned_objects:
            aligned_pcd += aligned_env + aligned_objects

        return aligned_env, aligned_objects

    def _align_environment(self, start_frame: int, end_frame: int) -> o3d.geometry.PointCloud:
        """
        Aligns the environment point clouds between the specified start and end frames.
        
        Processes each frame by loading the point cloud, down sampling, removing objects, 
        applying transformations, and aggregating the aligned point clouds.
        
        Args:
            start_frame (int): The starting frame index for environment alignment.
            end_frame (int): The ending frame index for environment alignment.
        
        Returns:
            o3d.geometry.PointCloud: The aggregated aligned environment point cloud.
        
        Raises:
            Exception: If there is an error during point cloud alignment.
        """
        self.logger.info("Starting Odometry Environment Alignment Process")

        frame_indices = list(np.arange(start_frame, end_frame + 1, self.frame_step))
        self.logger.info(f"Aligning frames from {start_frame} to {end_frame}: {frame_indices}")

        environment_pcd_queue = []
        for frame_index in frame_indices:
            self.logger.info(f"Processing frame {frame_index}")
            pcd = self.loki.load_ply(frame_index)
            odom = self.loki.load_odometry(frame_index)
            label3d = self.loki.load_label3d(frame_index)

            pcd = pcd.voxel_down_sample(voxel_size=0.05)  # TODO: pass this value
            pcd = self._remove_objects_from_environment(pcd, label3d)

            transformation_matrix = self.get_transformation_matrix(odom)
            pcd.transform(transformation_matrix)
            environment_pcd_queue.append(pcd)

        self.logger.info(f"Aligning {len(environment_pcd_queue)} point clouds")
        aligned_environment = o3d.geometry.PointCloud()
        for pcd in environment_pcd_queue:
            try:
                aligned_environment += pcd
            except Exception as e:
                self.logger.error(f"Error aligning point cloud: {e}")
                raise e

        self.logger.info("Aligned environment point clouds using odometry successfully")

        return aligned_environment

    def _align_last_objects_instance(self, end_frame: int) -> o3d.geometry.PointCloud:
        """
        Aligns and crops objects from the last frame.
        
        Loads the point cloud, odometry, and label3D data for the specified end frame, 
        filters for required object labels, crops the objects, applies transformations, 
        and aggregates the aligned objects.
        
        Args:
            end_frame (int): The frame index from which to align objects.
        
        Returns:
            o3d.geometry.PointCloud: The aggregated aligned objects point cloud.
        """
        self.logger.info("Starting Last Objects Instances Alignment Process")
        self.logger.info(f"Cropping objects from last frame: {end_frame}")

        pcd, odom, label3d = self.loki.load_alignment_data(end_frame)
        objects_needed = ['Car', 'Pedestrian']  # TODO: pass this value
        label3d = label3d[label3d['labels'].isin(objects_needed)]

        positions = label3d[['pos_x', 'pos_y', 'pos_z']].values
        dimensions = label3d[['dim_x', 'dim_y', 'dim_z']].values
        yaws = label3d['yaw'].values

        objects = [
            PointCloudUtils.crop_pcd(pcd, pos, dims, yaw)
            for pos, dims, yaw in zip(positions, dimensions, yaws)
        ]

        aligned_objects = o3d.geometry.PointCloud()
        for obj in objects:
            if obj:
                aligned_objects += obj

        transformation_matrix = self.get_transformation_matrix(odom)
        aligned_objects.transform(transformation_matrix)

        self.logger.info("Cropped objects from last point clouds successfully")

        return aligned_objects

    def _remove_objects_from_environment(self, pcd: o3d.geometry.PointCloud, label3d_df: pd.DataFrame) -> o3d.geometry.PointCloud:
        """
        Removes specified objects from the environment point cloud based on label3D data.
        
        Crops the point cloud around each object's position, dimensions, and yaw, 
        effectively removing the objects from the environment.
        
        Args:
            pcd (o3d.geometry.PointCloud): The original environment point cloud.
            label3d_df (pd.DataFrame): DataFrame containing label3D information for objects.
        
        Returns:
            o3d.geometry.PointCloud: The environment point cloud with specified objects removed.
        """
        positions = label3d_df[['pos_x', 'pos_y', 'pos_z']].values
        dimensions = label3d_df[['dim_x', 'dim_y', 'dim_z']].values
        yaws = label3d_df['yaw'].values

        count = 0
        for pos, dims, yaw in zip(positions, dimensions, yaws):
            pcd = PointCloudUtils.crop_pcd(pcd, pos, dims, yaw, remove=True)
            count += 1

        self.logger.debug(f"Removed {count} objects from pcd")
        
        return pcd

    def _get_start_end_frames(self, key_frame: int, align_interval: int, align_direction: AlignDirection) -> Tuple[int, int]:
        """
        Determines the start and end frame indices for alignment based on direction.
        
        Adjusts the alignment interval to account for even-only frame selection and 
        ensures the frame indices are correctly aligned.
        
        Args:
            key_frame (int): The reference frame index for alignment.
            align_interval (int): Number of frames to align on either side of the key frame.
            align_direction (AlignDirection): Direction of alignment (LEFT, RIGHT, SPLIT).
        
        Returns:
            Tuple[int, int]: A tuple containing the start and end frame indices.
        
        Raises:
            ValueError: If an invalid alignment direction is specified.
        """
        align_interval *= 2  # Double alignment to account for even-only frames

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

        # Adjust start_frame to the next even frame if it's odd
        if start_frame % 2 != 0:
            self.logger.warning(f"Start frame {start_frame} is odd. Adjusting to next even frame.")
            start_frame += 1

        # Adjust end_frame to the previous even frame if it's odd
        if end_frame % 2 != 0:
            self.logger.warning(f"End frame {end_frame} is odd. Adjusting to previous even frame.")
            end_frame -= 1

        return start_frame, end_frame
