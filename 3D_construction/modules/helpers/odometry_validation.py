from .align_direction import AlignDirection

class OdometryValidation:
    """
    A utility class for validating odometry alignment parameters.

    This class provides static methods to validate and adjust the key frame,
    alignment interval, and alignment direction for odometry-based processing.
    """

    @staticmethod
    def validate_alignment_input(key_frame: int, align_interval: int, align_direction: AlignDirection, max_frames: int, logger):
        """
        Validates the alignment input parameters.
        
        Ensures that the key frame is valid, the alignment interval is appropriate, 
        and there are enough frames in the specified direction for alignment.
        
        Args:
            key_frame (int): The reference frame index for alignment.
            align_interval (int): Number of frames to align on either side of the key frame.
            align_direction (AlignDirection): Direction of alignment (LEFT, RIGHT, SPLIT).
            max_frames (int): The total number of frames available for processing.
            logger: Logger instance for logging information and errors.
        
        Raises:
            ValueError: If any of the validation checks fail.
        """
        OdometryValidation.validate_key_frame(key_frame, logger)
        OdometryValidation.validate_align_interval(align_interval, logger)
        OdometryValidation.validate_align_direction(key_frame, align_interval, align_direction, max_frames, logger)

    @staticmethod
    def validate_key_frame(key_frame: int, logger):
        """
        Validates and adjusts the key frame to ensure it is positive and even.

        If the key frame is odd, it is decremented by one to make it even.
        Raises an error if the adjusted key frame becomes negative.

        Args:
            key_frame (int): The reference frame index for alignment.
            logger: Logger instance for logging information and errors.

        Raises:
            ValueError: If the key frame is negative after adjustment.
        """
        # Ensure the key frame is even
        if key_frame % 2 != 0:
            original_key_frame = key_frame
            key_frame -= 1
            logger.info(f"Key frame adjusted from {original_key_frame} to {key_frame} to ensure it is even.")

        # Validate that the key frame is positive
        if key_frame < 0:
            logger.error(f"Key frame {key_frame} is negative.")
            raise ValueError("Key frame must be a positive integer.")

    @staticmethod
    def validate_align_interval(align_interval: int, logger):
        """
        Validates the alignment interval to ensure it is positive and sufficient.

        The alignment interval must be a positive integer and allow at least one frame.

        Args:
            align_interval (int): The number of frames to align on either side of the key frame.
            logger: Logger instance for logging information and errors.

        Raises:
            ValueError: If the alignment interval is not positive or is too small to allow alignment.
        """
        # Ensure that the alignment interval allows at least one frame
        if align_interval <= 0:
            logger.error(f"Alignment interval must be positive. Given: {align_interval}")
            raise ValueError("Alignment interval must be a positive integer.")
        half_interval = align_interval // 2
        if half_interval < 1:
            logger.error(f"Alignment interval is too small: {half_interval} frame(s).")
            raise ValueError("Alignment interval must allow at least 1 frame.")

    @staticmethod
    def validate_align_direction(key_frame: int, align_interval: int, align_direction: AlignDirection, max_frames: int, logger):
        """
        Validates that there are enough frames in the specified alignment direction.

        Ensures the required number of frames exist in the chosen direction
        (LEFT, RIGHT, or SPLIT) relative to the key frame and alignment interval.

        Args:
            key_frame (int): The reference frame index for alignment.
            align_interval (int): Number of frames to align on either side of the key frame.
            align_direction (AlignDirection): Direction of alignment (LEFT, RIGHT, SPLIT).
            max_frames (int): The total number of frames available for processing.
            logger: Logger instance for logging information and errors.

        Raises:
            ValueError: If there are not enough frames in the specified direction,
                        or if an invalid alignment direction is provided.
        """
        original_align_interval = align_interval
        align_interval *= 2  # Convert to absolute number of frames
        half_interval = align_interval // 2

        if align_direction == AlignDirection.LEFT:
            required_start_frame = key_frame - align_interval
            if required_start_frame < 0:
                logger.error(
                    f"Not enough frames to the left of key frame {key_frame} with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames to the left for alignment.")

        elif align_direction == AlignDirection.RIGHT:
            required_end_frame = key_frame + align_interval
            if required_end_frame >= max_frames:
                logger.error(
                    f"Not enough frames to the right of key frame {key_frame} with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames to the right for alignment.")

        elif align_direction == AlignDirection.SPLIT:
            required_start_frame = key_frame - half_interval
            required_end_frame = key_frame + half_interval
            if required_start_frame < 0 or required_end_frame >= max_frames:
                logger.error(
                    f"Not enough frames on either side of key frame {key_frame} with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames on either side for split alignment.")

        else:
            logger.error(f"Invalid alignment direction: {align_direction}")
            raise ValueError("Invalid alignment direction specified.")
