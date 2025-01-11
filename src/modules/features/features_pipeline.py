import os

from modules.config.paths_loader import PathsLoader
from modules.features.extractors.pedestrian_movement_features import PedestrianMovementFeatures
from modules.features.processors.intention_binarizer import IntentionBinarizer
from modules.utilities.logger import LoggerUtils


def run_intention_binarizer_pipeline():
    """
    Main function to orchestrate the data preprocessing and splitting pipeline.
    """

    # Initialize logger
    logger = LoggerUtils.get_logger(__name__)

    # Define dataset paths
    loki_path = PathsLoader.get_folder_path(PathsLoader.Paths.LOKI)
    loki_csv_path = os.path.join(loki_path, "loki.csv")
    output_csv_path = os.path.join(loki_path, "binloki.csv")

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
    LoggerUtils.configure_unified_logging_file("logs/pedestrian_scenario_pipeline.log")
    logger = LoggerUtils.get_logger(__name__)

    # Define dataset and output directories
    dataset_directory = PathsLoader.get_folder_path(PathsLoader.Paths.RAW_DATA)
    output_directory = os.path.join(PathsLoader.get_folder_path(PathsLoader.Paths.PROCESSED_DATA),
                                    "features_group_walking")

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


if __name__ == "__main__":
    run_intention_binarizer_pipeline()
    run_pedestrian_movement_features_pipeline()
