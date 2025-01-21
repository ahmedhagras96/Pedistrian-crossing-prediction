import math
import os
from typing import Dict, Tuple

from modules.config.logger import Logger
from modules.features.utilities.utils import parse_odometry, parse_labels, calculate_speed_distance_movement


class SpeedDistanceScenarioProcessor:
    """
    Process a single scenario to extract speed, distance, and movement status features for each pedestrian.

    Args:
        scenario_path (str): Path to the scenario directory.
        fps (float): Frames per second used for speed calculation.
    """

    def __init__(self, scenario_path: str, fps: float) -> None:
        self.scenario_path = scenario_path
        self.fps = fps
        self.previous_positions: Dict[str, Tuple[float, float]] = {}
        self.prev_ego_position: Dict[str, float] = {}
        self.scenario_features: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.logger = Logger.get_logger(self.__class__.__name__)

    def process(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Process the scenario and compute features for each frame and pedestrian.

        Returns:
            dict: Nested dictionary mapping frame IDs to pedestrian feature dictionaries.
        """
        frame_files = sorted(os.listdir(self.scenario_path))
        odom_files = [f for f in frame_files if f.startswith("odom")]
        label_files = [f for f in frame_files if f.startswith("label3d")]

        for odom_file, label_file in zip(odom_files, label_files):
            self.logger.info(f"Processing odom file: {odom_file}")
            odom_frame_id = int(odom_file.split("_")[1].split(".")[0].split()[0])
            label_frame_id = int(label_file.split("_")[1].split(".")[0].split()[0])

            # Validate that the frame IDs match
            assert odom_frame_id == label_frame_id, (
                f"Frame mismatch: {odom_file} and {label_file} do not match."
            )

            odom_path = os.path.join(self.scenario_path, odom_file)
            label_path = os.path.join(self.scenario_path, label_file)

            ego_position = parse_odometry(odom_path)
            pedestrians = parse_labels(label_path)

            frame_features: Dict[str, Dict[str, float]] = {}
            self.logger.info(f"Pedestrians in frame {odom_frame_id}: {pedestrians}")

            for ped_id, ped_data in pedestrians.items():
                ped_x, ped_y = ped_data["x"], ped_data["y"]

                if ped_id in self.previous_positions:
                    prev_position = self.previous_positions[ped_id]
                    speed, distance, movement_status = calculate_speed_distance_movement(
                        prev_position, (ped_x, ped_y),
                        self.prev_ego_position, ego_position,
                        self.fps
                    )
                else:
                    speed = 0.0
                    distance = math.sqrt(ped_x ** 2 + ped_y ** 2)
                    movement_status = -1  # Unknown

                self.previous_positions[ped_id] = (ped_x, ped_y)
                frame_features[ped_id] = {
                    "speed": speed,
                    "distance": distance,
                    "movement_status": movement_status
                }

            self.scenario_features[odom_frame_id] = frame_features
            self.prev_ego_position = ego_position

        return self.scenario_features
