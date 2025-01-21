import json
import os
from typing import Dict

from modules.config.logger import Logger


class FeatureSaver:
    """
    Utility class to save features for each pedestrian to JSON files.
    """

    logger = Logger.get_logger("FeatureSaver")

    @staticmethod
    def save_features_per_pedestrian(
            scenario_id: str,
            group_status: Dict[int, Dict[str, int]],
            walking_status: Dict[int, Dict[str, int]],
            speed_distance_feas: Dict[int, Dict[str, Dict[str, float]]],
            output_folder: str,
            pid_avatars_dir: str
    ) -> None:
        """
        Save features per pedestrian for each frame in a scenario to JSON files.

        Args:
            scenario_id (str): Identifier for the scenario.
            group_status (dict): Group status data for pedestrians.
            walking_status (dict): Walking-toward-vehicle status for pedestrians.
            speed_distance_feas (dict): Speed, distance, and movement status features.
            output_folder (str): Directory to save output JSON files.
            pid_avatars_dir (str): Directory containing pedestrian avatar files for filtering.
        """
        avatar_ply_files = [f for f in os.listdir(pid_avatars_dir) if f.endswith('.ply')]
        filtered_peds = [os.path.splitext(f)[0] for f in avatar_ply_files]
        FeatureSaver.logger.info("Filtered pedestrians: %s", filtered_peds)

        for frame_id, ped_groups in group_status.items():
            for ped_id, group_value in ped_groups.items():
                zeros_frame_id = (4 - len(str(frame_id))) * "0"
                ped_file_base = f"{scenario_id.split('_')[1]}_{zeros_frame_id}{frame_id}_ped_{ped_id}"
                if ped_file_base not in filtered_peds:
                    FeatureSaver.logger.info("%s not in filtered pedestrians", ped_file_base)
                    continue

                walking_value = walking_status.get(frame_id, {}).get(ped_id, 0)
                pedestrian_features = {
                    "frame_id": frame_id,
                    "group_status": group_value,
                    "walking_toward_vehicle": walking_value
                }

                # Update with speed, distance, and movement status features
                if frame_id in speed_distance_feas and ped_id in speed_distance_feas[frame_id]:
                    pedestrian_features.update(speed_distance_feas[frame_id][ped_id])

                output_file = os.path.join(output_folder, f"{ped_file_base}.json")
                with open(output_file, 'w') as f:
                    json.dump(pedestrian_features, f, indent=4)
