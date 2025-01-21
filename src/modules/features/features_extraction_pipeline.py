from pathlib import Path

from modules.config.logger import Logger
from modules.config.paths_loader import PATHS
from modules.features.attention_vectors.features_processing_pipeline import features_processing_pipeline
from modules.features.avatar.avatar_feature_extraction import avatar_feature_extraction_by_scenario_pipeline
from modules.features.pedestrian_features.pedestrian_features_pipeline import pedestrian_features_pipeline
from modules.features.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline

#: Module-level logger for this script
logger = Logger.get_logger("FeatureExtractionPipeline")

if __name__ == '__main__':
    # Configure unified logging to write to a file for debugging or auditing
    Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("features_extraction_pipeline.log"))

    logger.debug("Entering main function.")

    run_preprocessing_pipeline()

    avatar_feature_extraction_by_scenario_pipeline()

    features_processing_pipeline()

    pedestrian_features_pipeline()

    logger.debug("Exiting main function.")
