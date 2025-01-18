import os

from modules.config.logger import LoggerUtils
from modules.config.paths_loader import PATHS
from modules.features.extractors.group_features import GroupFeatures
from modules.features.extractors.pedestrian_movement_features import PedestrianMovementFeatures
from modules.features.loaders.scenario_loader import ScenarioLoader
from modules.features.processors.intention_binarizer import IntentionBinarizer
from modules.utilities.scenario_utils import ScenarioUtils


def run_intention_binarizer_pipeline():
    """
    Main function to orchestrate the data preprocessing and splitting pipeline.
    """

    # Initialize logger
    logger = LoggerUtils.get_logger("IntentionBinarizerPipeline")

    # Define dataset paths
    loki_csv_path = PATHS.LOKI_CSV_PATH
    output_csv_path = PATHS.BIN_LOKI_CSV_PATH

    try:
        # Step 1: Load and preprocess the dataset
        logger.info("Step 1: Loading and preprocessing the dataset")
        data_loader = IntentionBinarizer()
        df = data_loader.load_and_binarize_intention(loki_csv_path)

        # Step 3: Save preprocessed DataFrame
        logger.info(f"Saving preprocessed dataset to {output_csv_path}")
        df.to_csv(output_csv_path, index=False)

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        raise


def run_pedestrian_movement_features_pipeline():
    """
    Main function to orchestrate the pedestrian scenario processing pipeline.
    """

    # Initialize logger
    logger = LoggerUtils.get_logger("PedestrianMovementFeaturesPipeline")

    # Define dataset and output directories
    dataset_directory = PATHS.RAW_DATA
    output_directory = os.path.join(PATHS.FEATURES, "features_group_walking")

    try:
        # Initialize processor
        logger.info("Initializing PedestrianScenarioProcessor")
        processor = PedestrianMovementFeatures(frames_per_second=5)

        # Process all scenarios
        logger.info("Starting pedestrian scenario processing")
        processor.process_all_pedestrian_scenarios(dataset_directory, output_directory)

        logger.info("Pedestrian scenario processing completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during scenario processing: {e}")
        raise


def extract_pedestrian_features():
    """
    Orchestrate extraction of pedestrian features from dataset.
    """
    # Initialize logger
    logger = LoggerUtils.get_logger("ExtractPedestrianFeaturesPipeline")

    dataset_folder = PATHS.RAW_DATA
    output_folder = os.path.join(PATHS.FEATURES, "pedestrian_features")
    ped_avatar_dir = os.path.join(PATHS.PROCESSED_DATA, "saved_pedestrians")

    logger.info("Starting pedestrian feature extraction.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output folder at {output_folder}")

    pmf = PedestrianMovementFeatures(frames_per_second=5)

    for scenario_name in sorted(os.listdir(dataset_folder)):
        scenario_path = os.path.join(dataset_folder, scenario_name)
        if not os.path.isdir(scenario_path):
            continue

        logger.info(f"Processing scenario: {scenario_name}...")

        # Load positions
        scenario_loader = ScenarioLoader()
        pedestrian_positions_frames = scenario_loader.load_label3d_positions(scenario_path)
        vehicle_positions = scenario_loader.load_vehicle_positions(scenario_path)

        # Compute group and walking statuses
        group_features = GroupFeatures()
        group_status = group_features.compute_group_status(pedestrian_positions_frames)
        walking_status = group_features.calculate_walking_toward_vehicle(pedestrian_positions_frames, vehicle_positions)

        # Extract speed/distance/movement status features
        speed_distance_features = pmf.get_speed_dist_ms_for_scenario(scenario_path, 5)

        # Save features per pedestrian
        scenario_utils = ScenarioUtils()
        scenario_utils.save_features_per_pedestrian(
            scenario_name,
            group_status,
            walking_status,
            speed_distance_features,
            output_folder,
            ped_avatar_dir
        )
    logger.info("Pedestrian feature extraction completed.")


if __name__ == "__main__":
    run_intention_binarizer_pipeline()
    run_pedestrian_movement_features_pipeline()
    extract_pedestrian_features()
