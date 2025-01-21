import os

from modules.config.logger import Logger
from modules.features.pedestrian_features.features.feature_saver import FeatureSaver
from modules.features.pedestrian_features.features.group_calculator import GroupCalculator
from modules.features.pedestrian_features.features.position_loader import PositionLoader
from modules.features.pedestrian_features.features.speed_distance_processor import SpeedDistanceScenarioProcessor


class PedestrianFeatureExtractor:
    """
    Orchestrates the extraction of pedestrian features across all scenarios.
    """

    def __init__(self, dataset_folder: str, output_folder: str, ped_avatar_dir: str, fps: float = 5) -> None:
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.ped_avatar_dir = ped_avatar_dir
        self.fps = fps

        self.logger = Logger.get_logger(self.__class__.__name__)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_features(self) -> None:
        """
        Process all scenarios to extract and save pedestrian features.
        """
        for scenario_name in sorted(os.listdir(self.dataset_folder)):
            scenario_path = os.path.join(self.dataset_folder, scenario_name)
            if not os.path.isdir(scenario_path):
                continue

            self.logger.info("Processing scenario: %s...", scenario_name)
            pedestrian_positions = PositionLoader.load_label3d_positions(scenario_path)
            vehicle_positions = PositionLoader.load_vehicle_positions(scenario_path)

            group_status = GroupCalculator.compute_group_status(pedestrian_positions)
            walking_status = GroupCalculator.calculate_walking_toward_vehicle(pedestrian_positions, vehicle_positions)

            speed_distance_processor = SpeedDistanceScenarioProcessor(scenario_path, self.fps)
            speed_distance_feas = speed_distance_processor.process()

            FeatureSaver.save_features_per_pedestrian(
                scenario_name, group_status, walking_status,
                speed_distance_feas, self.output_folder, self.ped_avatar_dir
            )
