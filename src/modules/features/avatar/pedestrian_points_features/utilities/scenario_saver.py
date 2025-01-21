import json
import os

from modules.config.logger import Logger


class ScenarioSaver:
    """
    A utility class to save extracted features in JSON format, organized by scenario.
    """

    _logger = Logger.get_logger("ScenarioSaver")

    @staticmethod
    def save_features_by_scenario(features: dict, output_directory: str) -> None:
        """
        Save extracted features to individual JSON files for each scenario.

        Args:
            features (dict): Dictionary of pedestrian IDs (file names) and their extracted features.
            output_directory (str): Path to the directory to save the scenario JSON files.

        Raises:
            OSError: If the directory cannot be created (e.g., permission issues).
        """
        ScenarioSaver._logger.info(f"Saving features by scenario to {output_directory}")
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        scenario_dict = {}

        # Organize features by scenario
        for file_name, feature_vector in features.items():
            base_name = os.path.basename(file_name)

            # Example naming convention: scenario_frame format => "ABC123_0001.ply"
            # Adjust parsing logic if your naming scheme differs
            scenario_id = base_name.split('_')[0]  # The prefix before the first underscore
            frame_number = base_name.split('_')[1]  # The part after the underscore

            if scenario_id not in scenario_dict:
                scenario_dict[scenario_id] = {}

            scenario_dict[scenario_id][f"frame_{frame_number}"] = feature_vector.tolist()

        # Save each scenario's data into separate JSON files
        for scenario_id, frames_data in scenario_dict.items():
            output_file = os.path.join(output_directory, f'scenario_{scenario_id}.json')
            with open(output_file, 'w') as json_file:
                json.dump(frames_data, json_file, indent=4)
            ScenarioSaver._logger.debug(f"Scenario {scenario_id} saved to {output_file}")
