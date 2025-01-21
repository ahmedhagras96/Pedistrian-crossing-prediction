from pathlib import Path

from modules.config.logger import Logger
from modules.config.paths_loader import PATHS
from modules.features.avatar.pedestrian_points_features.pedestrian_feature_extractor import PedestrianPointNetFeatureExtractor
from modules.features.avatar.pedestrian_points_features.scenario_feature_extractor import ScenarioFeatureExtractor

#: Module-level logger for this script
logger = Logger.get_logger("AvatarFeatureExtraction")


def avatar_feature_extraction_by_scenario_pipeline() -> None:
    """
    Extract features for pedestrian avatars by scenario.

    This function:
      1. Initializes a PointNet model (and optionally loads weights).
      2. Calls the ScenarioFeatureExtractor to process .ply files.
      3. Saves extracted features in JSON format, organized by scenario.

    Raises:
        FileNotFoundError: If any of the .ply files or model weights 
            (if loading them) are not found.
        RuntimeError: For other issues during the extraction or saving.
    """
    logger.info("Starting the pedestrian avatar feature extraction pipeline...")

    # Initialize the model
    model = PedestrianPointNetFeatureExtractor(input_dim=3, output_dim=64)
    # If you have a saved model, you can load it here:
    # logger.info("Loading model weights from path_to_weights.pt")
    # model.load_state_dict(torch.load("path_to_weights.pt"))
    model.eval()
    logger.debug("Model initialized and set to evaluation mode.")

    # Extract and save features by scenario
    ScenarioFeatureExtractor.extract_and_save_features_by_scenario(
        ply_folder=PATHS.SAVED_PEDESTRIANS_PATH,
        model=model,
        batch_size=16,
        output_directory=PATHS.PEDESTRIAN_AVATARS_FEATURES_PATH
    )

    logger.info("Pedestrian avatar feature extraction pipeline completed successfully.")


if __name__ == '__main__':
    """
    Main entry point for running the avatar feature extraction.
    
    This function calls the entire pipeline that:
      - Loads a PointNet model,
      - Processes pedestrian .ply files,
      - Saves extracted features in scenario-organized JSON files.
    
    Raises:
        RuntimeError: If any part of the pipeline fails.
    """
    # Configure unified logging to write to a file for debugging or auditing
    Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("avatar_feature_extraction.log"))

    logger.debug("Entering main function.")
    avatar_feature_extraction_by_scenario_pipeline()
    logger.debug("Exiting main function.")
