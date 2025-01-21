from modules.features.avatar.pedestrian_points_features.feature_batch_extractor import FeatureBatchExtractor
from modules.features.avatar.pedestrian_points_features.pedestrian_feature_extractor import PedestrianPointNetFeatureExtractor
from modules.features.avatar.pedestrian_points_features.utilities.scenario_saver import ScenarioSaver
from modules.config.logger import Logger


class ScenarioFeatureExtractor:
    """
    A high-level orchestrator class that extracts features from .ply files in batches
    and saves them in JSON format, organized by scenario.
    """

    _logger = Logger.get_logger("ScenarioFeatureExtractor")

    @staticmethod
    def extract_and_save_features_by_scenario(
            ply_folder: str,
            model: PedestrianPointNetFeatureExtractor,
            batch_size: int,
            output_directory: str
    ) -> None:
        """
        Extract features from pedestrian .ply files using PointNet and save by scenario.

        Args:
            ply_folder (str): Path to the folder containing .ply files.
            model (PointNetFeatureExtractor): A trained PointNet model.
            batch_size (int): Number of pedestrians per batch.
            output_directory (str): Directory to save the scenario JSON files.
        """
        ScenarioFeatureExtractor._logger.info(
            f"Starting scenario-based feature extraction from {ply_folder}"
        )

        # Extract features in batches
        features = FeatureBatchExtractor.extract_features_in_batches(
            ply_folder=ply_folder,
            model=model,
            batch_size=batch_size
        )

        # Save features by scenario
        ScenarioSaver.save_features_by_scenario(
            features=features,
            output_directory=output_directory
        )

        ScenarioFeatureExtractor._logger.info("Scenario-based feature extraction completed.")
