import os

from modules.base_module import BaseModule
from modules.utilities.file_utils import FileUtils


class ScenarioUtils(BaseModule):
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
                FileUtils.save_json(frames_data, output_file)
                self.logger.info(f"Scenario {scenario_id} features saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving scenario {scenario_id} to {output_file}: {e}")

    def save_features_per_pedestrian(self, scenario_id, group_status, walking_status, scenario_features, output_folder,
                                     pid_avatars_dir):
        """
        Save features per pedestrian to JSON files.
    
        Args:
            scenario_id (str): ID of the scenario.
            group_status (dict): Group status per frame.
            walking_status (dict): Walking toward vehicle status per frame.
            scenario_features (dict): Extracted speed/distance/movement features.
            output_folder (str): Directory to save JSON files.
            pid_avatars_dir (str): Directory containing pedestrian avatar .ply files.
        """
        self.logger.info(f"Saving features per pedestrian for scenario {scenario_id}")
        avatar_ply_files = [f for f in os.listdir(pid_avatars_dir) if f.endswith('.ply')]
        filtered_peds = [os.path.splitext(f)[0] for f in avatar_ply_files]
        self.logger.debug(f"Filtered pedestrians: {filtered_peds}")

        for frame_id in group_status.keys():
            for ped_id, group_value in group_status[frame_id].items():
                walking_value = walking_status.get(frame_id, {}).get(ped_id, 0)

                pedestrian_features = {
                    "frame_id": frame_id,
                    "group_status": group_value,
                    "walking_toward_vehicle": walking_value
                }

                if frame_id in scenario_features and ped_id in scenario_features[frame_id]:
                    pedestrian_features.update(scenario_features[frame_id][ped_id])

                zeros_frame_id = (4 - len(str(frame_id))) * "0"
                pedistrian_file_name = f"{scenario_id.split('_')[1]}_{zeros_frame_id}{frame_id}_ped_{ped_id}"

                if pedistrian_file_name not in filtered_peds:
                    self.logger.debug(f"{pedistrian_file_name} not in filtered pedestrians")
                    continue

                output_file = os.path.join(output_folder, f"{pedistrian_file_name}.json")

                FileUtils.save_json(pedestrian_features, output_file)
                self.logger.info(f"Saved features for pedestrian {ped_id} at frame {frame_id} to {output_file}")
