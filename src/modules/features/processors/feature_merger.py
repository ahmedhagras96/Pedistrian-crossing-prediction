import os

from modules.config.logger import LoggerUtils
from modules.utilities.file_utils import FileUtils


class FeatureMerger:
    """
    A utility class for merging pedestrian features from different JSON files.
    """

    def __init__(self):
        """Initialize the FeatureMerger and set up logging."""
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

    def merge_features(self, group_walking_file, speed_distance_file, output_file):
        """
        Merge two JSON files for a scenario based on pedestrian IDs.

        Args:
            group_walking_file (str): Path to the group/walking JSON file.
            speed_distance_file (str): Path to the speed/distance JSON file.
            output_file (str): Path to save the merged JSON file.
        """
        try:
            self.logger.info(f"Merging features: {group_walking_file}, {speed_distance_file}")

            # Load JSON files
            group_walking_data = FileUtils.load_json(group_walking_file)
            speed_distance_data = FileUtils.load_json(speed_distance_file)

            # Initialize merged data
            merged_data = {}

            # Iterate through frames in both files
            for frame_id, ped_data_gw in group_walking_data.items():
                if frame_id in speed_distance_data:
                    ped_data_sd = speed_distance_data[frame_id]
                    merged_frame = {}

                    # Match pedestrian IDs
                    for ped_id, features_gw in ped_data_gw.items():
                        if ped_id in ped_data_sd:
                            features_sd = ped_data_sd[ped_id]

                            # Concatenate features
                            merged_frame[ped_id] = {
                                "group_status": features_gw["group_status"],
                                "walking_toward_vehicle": features_gw["walking_toward_vehicle"],
                                "speed": features_sd["speed"],
                                "distance": features_sd["distance"],
                                "movement_status": features_sd["movement_status"]
                            }

                    # Add merged frame data
                    if merged_frame:
                        merged_data[frame_id] = merged_frame

            # Save merged data
            FileUtils.save_json(merged_data, output_file)

            self.logger.info(f"Merged features saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error merging features: {e}")
            raise e

    def merge_all_scenarios(self, group_walking_dir, speed_distance_dir, output_dir):
        """
        Merge features for all scenarios in the given directories.

        Args:
            group_walking_dir (str): Directory containing group/walking JSON files.
            speed_distance_dir (str): Directory containing speed/distance JSON files.
            output_dir (str): Directory to save merged JSON files.
        """
        self.logger.info("Starting feature merging for all scenarios")
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(group_walking_dir):
            if filename.endswith("_features.json"):
                scenario_name = filename.replace("_features.json", "")
                group_walking_file = os.path.join(group_walking_dir, filename)
                speed_distance_file = os.path.join(speed_distance_dir, f"{scenario_name}_features.json")
                output_file = os.path.join(output_dir, f"{scenario_name}_merged_features.json")

                if os.path.exists(speed_distance_file):
                    self.merge_features(group_walking_file, speed_distance_file, output_file)
                else:
                    self.logger.warning(f"Missing speed/distance file for scenario: {scenario_name}")
