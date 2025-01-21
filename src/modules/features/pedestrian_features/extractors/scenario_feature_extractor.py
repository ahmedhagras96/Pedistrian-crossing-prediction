import math
import os
from typing import Dict, Tuple

from modules.features.utilities.utils import parse_odometry, parse_labels, calculate_speed_distance_movement


class ScenarioFeatureExtractor:
    """
    Extracts speed, distance, and movement status features for pedestrians within a single scenario.

    Args:
        scenario_path (str): Path to the scenario directory containing odometry and label files.
        fps (float): Frames per second for calculating speed.
    """

    def __init__(self, scenario_path: str, fps: float) -> None:
        self.scenario_path = scenario_path
        self.fps = fps
        self.previous_positions: Dict[str, Tuple[float, float]] = {}
        self.pending_corrections: Dict[str, int] = {}
        self.prev_ego_position: Dict[str, float] = {}
        self.scenario_features: Dict[int, Dict[str, Dict[str, float]]] = {}

    def process(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Process the scenario to extract features for each pedestrian over frames.

        Returns:
            dict: Nested dictionary mapping frame IDs to pedestrian features 
                  (speed, distance, movement_status).
        """
        frame_files = sorted(os.listdir(self.scenario_path))
        odom_files = [f for f in frame_files if f.startswith("odom")]
        label_files = [f for f in frame_files if f.startswith("label3d")]

        for odom_file, label_file in zip(odom_files, label_files):
            # Extract frame identifier from filename
            cleaned_name = odom_file.split("_")[1].split(".")[0].split()[0]
            frame_id = int(cleaned_name)

            odom_path = os.path.join(self.scenario_path, odom_file)
            label_path = os.path.join(self.scenario_path, label_file)

            # Parse current odometry and label data
            ego_position = parse_odometry(odom_path)
            pedestrians = parse_labels(label_path)

            frame_features: Dict[str, Dict[str, float]] = {}

            for ped_id, ped_data in pedestrians.items():
                ped_x, ped_y = ped_data["x"], ped_data["y"]

                if ped_id in self.previous_positions:
                    prev_position = self.previous_positions[ped_id]
                    speed, distance, movement_status = calculate_speed_distance_movement(
                        prev_position, (ped_x, ped_y),
                        self.prev_ego_position, ego_position,
                        self.fps
                    )

                    if ped_id in self.pending_corrections:
                        pending_frame_id = self.pending_corrections[ped_id]
                        # Remove pending corrections from previously stored features
                        if pending_frame_id in self.scenario_features:
                            self.scenario_features[pending_frame_id].pop(ped_id, None)
                        del self.pending_corrections[ped_id]
                else:
                    speed = 0.0
                    distance = math.sqrt(ped_x ** 2 + ped_y ** 2)
                    movement_status = -1  # Unknown
                    self.pending_corrections[ped_id] = frame_id

                self.previous_positions[ped_id] = (ped_x, ped_y)
                frame_features[ped_id] = {
                    "speed": speed,
                    "distance": distance,
                    "movement_status": movement_status
                }

            self.scenario_features[frame_id] = frame_features
            self.prev_ego_position = ego_position

        return self.scenario_features
