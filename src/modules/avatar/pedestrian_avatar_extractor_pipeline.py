from pathlib import Path

from modules.avatar.extraction.avatar_extraction_pipeline import AvatarExtractionPipeline
from modules.config.logger import Logger
from modules.config.paths_loader import PATHS

#: Module-level logger for this script
logger = Logger.get_logger("PedestrianAvatarExtractor")


def pedestrian_avatar_extractor_pipeline() -> None:
    """
    Run the pipeline to process point cloud data for pedestrian avatars.

    This function:
      1. Instantiates a PedestrianProcessingPipeline with configuration parameters from PATHS.
      2. Loads scenario and frame IDs.
      3. Validates scenario and frame IDs.
      4. Processes valid frames and saves the cropped pedestrian point clouds.

    Raises:
        KeyboardInterrupt: If the user interrupts the operation manually.
        FileNotFoundError: If the specified CSV or root directory does not exist.
        RuntimeError: If any step in the pipeline fails unexpectedly.
    """
    logger.info("Starting pedestrian avatar extraction pipeline...")

    # Paths and constants
    root_dir = PATHS.SAMPLE_DATA_PATH
    csv_path = PATHS.BIN_LOKI_CSV_FILE
    threshold_multiplier = 0.5
    output_csv = PATHS.PEDESTRIAN_POINTCLOUDS_CSV_FILE  # Not used directly in this snippet
    save_directory = PATHS.SAVED_PEDESTRIANS_PATH

    logger.debug(f"Root dir: {root_dir}")
    logger.debug(f"CSV path: {csv_path}")
    logger.debug(f"Threshold multiplier: {threshold_multiplier}")
    logger.debug(f"Output CSV: {output_csv}")
    logger.debug(f"Save directory: {save_directory}")

    # Create the pipeline instance
    pipeline = AvatarExtractionPipeline(
        root_dir=root_dir,
        csv_path=csv_path,
        save_dir=save_directory,
        threshold_multiplier=threshold_multiplier
    )

    try:
        # 1. Load scenario and frame IDs
        scenario_ids, frame_ids = pipeline.load_scenario_frame_ids()
        logger.info(f"Loaded {len(scenario_ids)} scenario IDs and {len(frame_ids)} frame IDs.")

        # 2. Validate scenarios and frames
        valid_scenario_ids, valid_frame_ids = pipeline.validate_scenarios_and_frames(scenario_ids, frame_ids)
        logger.info(f"Validated scenario IDs: {valid_scenario_ids}")
        logger.info(f"Validated frame IDs: {valid_frame_ids}")

        # 3. Process all valid frames & save point clouds
        pipeline.process_all_frames_and_crop_pedestrians(valid_scenario_ids, valid_frame_ids)
        logger.info("All valid frames processed and pedestrian point clouds saved.")

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user. Exiting gracefully...")
        print("\nOperation cancelled by user. Exiting gracefully...")


if __name__ == '__main__':
    """
    Main entry point for running the pedestrian avatar extractor pipeline.
    
    This function calls the entire pipeline that:
      1. Loads and verifies scenario/frame data.
      2. Processes point clouds for pedestrians.
      3. Saves cropped pedestrian point clouds to disk.
    
    Raises:
        RuntimeError: If any part of the pipeline fails (not explicitly caught).
    """
    # Configure unified logging to write to a file
    Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("pedestrian_avatar_extractor.log"))

    logger.debug("Entering main function.")
    pedestrian_avatar_extractor_pipeline()
    logger.debug("Exiting main function.")
