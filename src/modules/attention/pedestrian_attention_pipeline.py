import os

from modules.attention.pedestrian_attention.pedestrain_attention_model import PedestrianAttentionModel
from modules.config.paths_loader import PathsLoader
from modules.features.features_processors.feature_merger import FeatureMerger
from modules.utilities.logger import LoggerUtils


def run_pedestrian_attention_pipeline():
    """
    Main function to orchestrate the pedestrian feature processing pipeline.
    """

    # Initialize logger
    logger = LoggerUtils.get_logger(__name__)

    # Directories for input and output
    processed_data_path = PathsLoader.get_folder_path(PathsLoader.Paths.PROCESSED_DATA)
    output_data_path = PathsLoader.get_folder_path(PathsLoader.Paths.OUTPUT)
    group_walking_dir = os.path.join(processed_data_path, "features_group_walking")
    speed_distance_dir = os.path.join(processed_data_path, "features_speed_distance")
    merged_output_dir = os.path.join(output_data_path, "features_merged")
    attention_output_dir = os.path.join(output_data_path, "attention_results")

    # Parameters for attention processing
    input_dim = 5  # Number of input features (group_status, walking_toward_vehicle, scaled_speed, scaled_distance, movement_status)
    num_heads = 5  # Number of attention heads

    try:
        # Step 1: Merge features from multiple sources
        logger.info("Step 1: Merging features from multiple sources")
        feature_merger = FeatureMerger()
        feature_merger.merge_all_scenarios(group_walking_dir, speed_distance_dir, merged_output_dir)

        # Step 2: Apply attention processing to the merged features
        logger.info("Step 2: Applying attention processing to the merged features")
        attention_processor = PedestrianAttentionModel(input_dim=input_dim, num_heads=num_heads)

        # Process each merged JSON file
        for json_file in os.listdir(merged_output_dir):
            if json_file.endswith(".json"):
                input_file = os.path.join(merged_output_dir, json_file)
                output_file = os.path.join(attention_output_dir, f"attention_{json_file}")
                attention_processor.process_scenario(input_file, output_file)

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        raise


if __name__ == "__main__":
    run_pedestrian_attention_pipeline()
