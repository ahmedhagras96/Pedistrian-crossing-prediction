import json
import os

from modules.config.logger import Logger


class FeaturesMerger:
    """
    A class to merge feature JSON files from different sources into a unified format.
    """
    logger = Logger.get_logger("FeaturesMerger")

    @classmethod
    def merge_features(cls, group_walking_file: str, speed_distance_file: str, output_file: str) -> None:
        """
        Merge two JSON files for a scenario based on pedestrian IDs and save the result.

        Args:
            group_walking_file (str): Path to the group walking JSON file.
            speed_distance_file (str): Path to the speed and distance JSON file.
            output_file (str): Path where the merged JSON should be saved.

        Returns:
            None
        """
        cls.logger.info(f"Merging features from {group_walking_file} and {speed_distance_file}")
        # Load JSON files
        with open(group_walking_file, "r") as gw_file:
            group_walking_data = json.load(gw_file)

        with open(speed_distance_file, "r") as sd_file:
            speed_distance_data = json.load(sd_file)

        merged_data = {}

        # Iterate through frames and merge data
        for frame_id, ped_data_gw in group_walking_data.items():
            if frame_id in speed_distance_data:
                ped_data_sd = speed_distance_data[frame_id]
                merged_frame = {}

                for ped_id, features_gw in ped_data_gw.items():
                    if ped_id in ped_data_sd:
                        features_sd = ped_data_sd[ped_id]
                        merged_frame[ped_id] = {
                            "group_status": features_gw.get("group_status"),
                            "walking_toward_vehicle": features_gw.get("walking_toward_vehicle"),
                            "speed": features_sd.get("speed"),
                            "distance": features_sd.get("distance"),
                            "movement_status": features_sd.get("movement_status")
                        }

                if merged_frame:
                    merged_data[frame_id] = merged_frame

        # Save merged data to output file
        with open(output_file, "w") as out_file:
            json.dump(merged_data, out_file, indent=4)

        cls.logger.info(f"Merged features saved to {output_file}")

    @classmethod
    def merge_all_scenarios_features(cls, group_walking_dir: str, speed_distance_dir: str, output_dir: str) -> None:
        """
        Process all scenarios by merging corresponding group/walking and speed/distance JSON files.

        Args:
            group_walking_dir (str): Directory containing group walking JSON files.
            speed_distance_dir (str): Directory containing speed and distance JSON files.
            output_dir (str): Directory where merged JSON files will be saved.

        Returns:
            None
        """
        cls.logger.info("Starting merge of all scenarios features")
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(group_walking_dir):
            if filename.endswith("_features.json"):
                scenario_name = filename.replace("_features.json", "")
                group_walking_file = os.path.join(group_walking_dir, filename)
                speed_distance_file = os.path.join(speed_distance_dir, f"{scenario_name}_features.json")
                output_file = os.path.join(output_dir, f"{scenario_name}_merged_features.json")

                if os.path.exists(speed_distance_file):
                    cls.merge_features(group_walking_file, speed_distance_file, output_file)
                else:
                    cls.logger.warning(f"Missing speed/distance file for scenario: {scenario_name}")
