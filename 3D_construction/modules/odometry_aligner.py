# modules/odometry_processor.py
from enum import Enum
from typing import Tuple
import numpy as np

from .base_aligner import BaseAligner
from .utils.logger import Logger

class AlignDirection(Enum):
    LEFT = 1
    RIGHT = 2
    SPLIT = 3

class PointCloudOdometryAligner(BaseAligner):
    """
    Processor for aligning point clouds based on vehicle odometry.
    """

    # Initialize a class-level logger
    logger = Logger.get_logger(__name__)
    
    def __init__(self, scenario_path: str, loki_csv_path: str, key_frame: int, max_frames: int = 100, frame_step: int = 2):
        super().__init__(scenario_path, loki_csv_path, key_frame, max_frames, frame_step)
        
        PointCloudOdometryAligner.logger.info(f"Initialized {self.__class__.__name__}")
        
    def align(self, key_frame: int = None, align_interval: int = 10, align_direction: AlignDirection = AlignDirection.SPLIT):
        PointCloudOdometryAligner.logger.info(f"Starting Odometry Alignment Process")
        
        if key_frame is None:
            key_frame = self.key_frame
        
        self._validate_alignment_input(key_frame, align_interval, align_direction)
        PointCloudOdometryAligner.logger.info(f"All input data valid")
        PointCloudOdometryAligner.logger.info(f"Key frame set to frame {key_frame}")
        PointCloudOdometryAligner.logger.info(f"Alingment Inerval set to {align_interval}")
        PointCloudOdometryAligner.logger.info(f"Alingment Direction set to {align_direction.name}")

        self._align_environment(key_frame, align_interval, align_direction)
    
    def _validate_alignment_input(self, key_frame:int,  align_interval: int, align_direction: AlignDirection):
        self._validate_key_frame()
        self._validate_align_interval(align_interval)
        self._validate_align_direction(key_frame, align_interval, align_direction)

    def _align_environment(self, key_frame: int, align_interval: int, align_direction: AlignDirection):
        PointCloudOdometryAligner.logger.info(f"Starting Odometry Environment Alignment Process")
        PointCloudOdometryAligner.logger.info(f"Aligning {align_interval} frames in {align_direction.name} direction")

        start_frame, end_frame = self._get_start_end_frames(key_frame, align_interval, align_direction)
        env_frame_indicies = list(np.arange(start_frame, end_frame + 1, self.frame_step))
        PointCloudOdometryAligner.logger.info(f"Aligning frames from {start_frame} to frames {end_frame}: {env_frame_indicies}")
        
        
    def _get_start_end_frames(self, key_frame: int, align_interval: int, align_direction: AlignDirection) -> Tuple[int, int]:
        start_frame = 0
        end_frame = 0
        
        # Double alignment to account for even only frames
        align_interval = align_interval * 2
        
        if align_direction == AlignDirection.LEFT:
            start_frame = key_frame - align_interval
            end_frame = key_frame
        if align_direction == AlignDirection.RIGHT:
            start_frame = key_frame
            end_frame = key_frame + align_interval
        if align_direction == AlignDirection.SPLIT:
            start_frame = key_frame - (align_interval // 2)
            end_frame = key_frame + (align_interval // 2)

        # Adjust start_frame to the next even frame if it's odd
        if start_frame % 2 != 0:
            self.logger.warning(f"Start frame {start_frame} is odd. Adjusting to next even frame.")
            start_frame = start_frame + 1
    
        # Adjust end_frame to the previous even frame if it's odd
        if end_frame % 2 != 0:
            self.logger.warning(f"End frame {end_frame} is odd. Adjusting to previous even frame.")
            end_frame = end_frame - 1
        
        return start_frame, end_frame

    def _validate_key_frame(self):
        # Ensure the key frame is even
        if self.key_frame % 2 != 0:
            original_key_frame = self.key_frame
            self.key_frame -= 1
            PointCloudOdometryAligner.logger.info(f"Key frame adjusted from {original_key_frame} to {self.key_frame} to ensure it is even.")

        # Validate that the key frame is positive
        if self.key_frame < 0:
            PointCloudOdometryAligner.logger.error(f"Key frame {self.key_frame} is negative.")
            raise ValueError("Key frame must be a positive integer.")

    def _validate_align_interval(self, align_interval: int):
        # Ensure that the alignment interval allows at least one frame
        half_interval = align_interval // 2
        if half_interval < 1 or align_interval < 0:
            self.logger.error(f"Alignment interval is too small: {half_interval} frame(s).")
            raise ValueError("Alignment interval after adjustment must allow at least 1 frame.")

    def _validate_align_direction(self, key_frame: int, align_interval: int, align_direction: AlignDirection):
        # Verify that there are enough frames in the specified alignment direction
        original_align_interval = align_interval
        align_interval = align_interval * 2
        half_interval = align_interval // 2

        if align_direction == AlignDirection.LEFT:
            required_start_frame = key_frame - align_interval
            if required_start_frame < 0:
                self.logger.error(
                    f"Not enough frames to the left of key frame {key_frame} with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames to the left for alignment.")

        elif align_direction == AlignDirection.RIGHT:
            required_end_frame = key_frame + align_interval
            if required_end_frame >= self.max_frames:
                self.logger.error(
                    f"Not enough frames to the right of key frame {self.key_frame} with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames to the right for alignment.")

        elif align_direction == AlignDirection.SPLIT:
            required_start_frame = key_frame - half_interval
            required_end_frame = key_frame + half_interval
            if required_start_frame < 0 or required_end_frame >= self.max_frames:
                self.logger.error(
                    f"Not enough frames on either side of key frame {key_frame} with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames on either side for split alignment.")
        else:
            self.logger.error(f"Invalid alignment direction: {align_direction}")
            raise ValueError("Invalid alignment direction specified.")
