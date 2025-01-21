from modules.config.logger import Logger
from modules.reconstruction.aligners.align_direction import AlignDirection


class LoggerUtils:
    pass


class OdometryValidation:
    """
    A utility class for validating odometry alignment parameters.

    This class provides static methods to validate and adjust the key frame,
    alignment interval, and alignment direction for odometry-based processing.
    """

    # Class-level logger for static methods
    _logger = Logger.get_logger("OdometryValidation")

    @staticmethod
    def validate_alignment_input(
            key_frame: int,
            align_interval: int,
            align_direction: AlignDirection,
            max_frames: int
    ) -> None:
        """
        Validate the alignment input parameters.

        Ensures that the key frame is valid, the alignment interval is appropriate,
        and there are enough frames in the specified direction for alignment.

        Args:
            key_frame (int): The reference frame index for alignment.
            align_interval (int): Number of frames to align on either side of the key frame.
            align_direction (AlignDirection): Direction of alignment (LEFT, RIGHT, SPLIT).
            max_frames (int): The total number of frames available for processing.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        OdometryValidation._logger.info("Validating alignment input parameters.")
        OdometryValidation.validate_key_frame(key_frame)
        OdometryValidation._logger.info("Key frame validated.")

        OdometryValidation.validate_align_interval(align_interval)
        OdometryValidation._logger.info("Alignment interval validated.")

        OdometryValidation.validate_align_direction(
            key_frame, align_interval, align_direction, max_frames
        )
        OdometryValidation._logger.info("Alignment direction validated.")

    @staticmethod
    def validate_key_frame(key_frame: int) -> None:
        """
        Validate and adjust the key frame to ensure it is positive and even.

        If the key frame is odd, it is decremented by one to make it even.
        Raises an error if the adjusted key frame becomes negative.

        Args:
            key_frame (int): The reference frame index for alignment.

        Raises:
            ValueError: If the key frame is negative after adjustment.
        """
        if key_frame % 2 != 0:
            original_key_frame = key_frame
            key_frame -= 1
            OdometryValidation._logger.info(
                f"Key frame adjusted from {original_key_frame} to {key_frame} to ensure it is even."
            )

        if key_frame < 0:
            OdometryValidation._logger.error(f"Key frame {key_frame} is negative.")
            raise ValueError("Key frame must be a positive integer.")

    @staticmethod
    def validate_align_interval(align_interval: int) -> None:
        """
        Validate the alignment interval to ensure it is positive and sufficient.

        The alignment interval must be a positive integer and allow at least one frame
        on each side (when considered in SPLIT mode).

        Args:
            align_interval (int): The number of frames to align around the key frame.

        Raises:
            ValueError: If the alignment interval is not positive or too small to allow alignment.
        """
        if align_interval <= 0:
            OdometryValidation._logger.error(
                f"Invalid alignment interval: {align_interval}. Must be positive."
            )
            raise ValueError("Alignment interval must be a positive integer.")

        half_interval = align_interval // 2
        if half_interval < 1:
            OdometryValidation._logger.error(
                f"Alignment interval is too small to allow at least one frame: {half_interval}."
            )
            raise ValueError("Alignment interval must allow at least one frame on each side.")

    @staticmethod
    def validate_align_direction(
            key_frame: int,
            align_interval: int,
            align_direction: AlignDirection,
            max_frames: int
    ) -> None:
        """
        Validate that there are enough frames in the specified alignment direction.

        Ensures the required number of frames exist in the chosen direction
        (LEFT, RIGHT, or SPLIT) relative to the key frame and alignment interval.

        Args:
            key_frame (int): The reference frame index for alignment.
            align_interval (int): Number of frames to align around the key frame.
            align_direction (AlignDirection): Direction of alignment (LEFT, RIGHT, SPLIT).
            max_frames (int): The total number of frames available for processing.

        Raises:
            ValueError: If there are not enough frames in the specified direction
                        or if an invalid alignment direction is provided.
        """
        original_align_interval = align_interval
        align_interval *= 2  # Convert to total frames needed in one direction
        half_interval = align_interval // 2

        if align_direction == AlignDirection.LEFT:
            required_start_frame = key_frame - align_interval
            if required_start_frame < 0:
                OdometryValidation._logger.error(
                    f"Not enough frames to the left of key frame {key_frame} "
                    f"with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames to the left for alignment.")

        elif align_direction == AlignDirection.RIGHT:
            required_end_frame = key_frame + align_interval
            if required_end_frame >= max_frames:
                OdometryValidation._logger.error(
                    f"Not enough frames to the right of key frame {key_frame} "
                    f"with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames to the right for alignment.")

        elif align_direction == AlignDirection.SPLIT:
            required_start_frame = key_frame - half_interval
            required_end_frame = key_frame + half_interval
            if (required_start_frame < 0) or (required_end_frame >= max_frames):
                OdometryValidation._logger.error(
                    f"Not enough frames on either side of key frame {key_frame} "
                    f"with alignment interval {original_align_interval}."
                )
                raise ValueError("Not enough frames on either side for split alignment.")

        else:
            OdometryValidation._logger.error(f"Invalid alignment direction specified: {align_direction}")
            raise ValueError("Invalid alignment direction specified.")
