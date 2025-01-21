import json
import os

from modules.config.logger import Logger
from modules.features.pedestrian_features.extractors.scenario_feature_extractor import ScenarioFeatureExtractor


class AllScenariosProcessor:
    """
    Processes all scenarios in a given root directory to extract and save features.

    Args:
        root_directory (str): Path to the root directory containing scenario subdirectories.
        output_directory (str): Path where the extracted features JSON files will be saved.
        fps (float): Frames per second for calculating speeds.
    """

    def __init__(self, root_directory: str, output_directory: str, fps: float) -> None:
        self.root_directory = root_directory
        self.output_directory = output_directory
        self.fps = fps
        os.makedirs(self.output_directory, exist_ok=True)
        
        self.logger = Logger.get_logger(self.__class__.__name__)

    def process_all(self) -> None:
        """
        Process all scenarios found in the root directory, extract features, 
        and save them as JSON files in the output directory.
        """
        for scenario in os.listdir(self.root_directory):
            scenario_path = os.path.join(self.root_directory, scenario)
            if not os.path.isdir(scenario_path):
                continue

            self.logger.info(f"Processing scenario: {scenario}...")
            extractor = ScenarioFeatureExtractor(scenario_path, self.fps)
            scenario_features = extractor.process()

            output_path = os.path.join(self.output_directory, f"{scenario}_features.json")
            with open(output_path, "w") as json_file:
                json.dump(scenario_features, json_file, indent=4)

            self.logger.info(f"Saved features for {scenario} to {output_path}")
