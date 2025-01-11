import json
import os

from modules.utilities.base_utility import BaseUtility


class ScenarioUtils(BaseUtility):
    """
    A utility class for grouping extracted features by scenario and saving them as JSON files.

    Methods:
        group_by_scenario(features: dict) -> dict:
            Organize features by scenario ID and frame number.

        save_to_json(grouped_features: dict, output_dir: str):
            Save grouped features into scenario-specific JSON files.
    """

    def __init__(self):
        super().__init__()

    def group_by_scenario(self, features):
        """
        Organize features by scenario ID and frame number.

        Args:
            features (dict): A dictionary mapping file names to feature vectors.

        Returns:
            dict: A dictionary of scenarios, each containing frame-specific feature vectors.
        """
        self.logger.info("Grouping features by scenario")
        scenario_dict = {}

        for file_name, feature_vector in features.items():
            base_name = os.path.basename(file_name)
            scenario_id, frame_number = base_name.split('_')[:2]

            if scenario_id not in scenario_dict:
                scenario_dict[scenario_id] = {}

            scenario_dict[scenario_id][f"frame_{frame_number}"] = feature_vector.tolist()

        self.logger.info(f"Grouping completed. Found {len(scenario_dict)} scenarios.")
        return scenario_dict

    def save_to_json(self, grouped_features, output_dir):
        """
        Save grouped features into scenario-specific JSON files.

        Args:
            grouped_features (dict): Features grouped by scenario.
            output_dir (str): Path to the directory for saving JSON files.
        """
        self.logger.info(f"Saving scenario features to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        for scenario_id, frames_data in grouped_features.items():
            output_file = os.path.join(output_dir, f"scenario_{scenario_id}.json")
            try:
                with open(output_file, 'w') as json_file:
                    json.dump(frames_data, json_file, indent=4)
                self.logger.info(f"Scenario {scenario_id} features saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving scenario {scenario_id} to {output_file}: {e}")
