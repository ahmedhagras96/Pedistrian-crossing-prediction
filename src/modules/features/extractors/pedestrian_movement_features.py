import json
import math
import os

from modules.utilities.logger import LoggerUtils
from modules.utilities.pedestrian_data_utils import PedestrianDataUtils


class PedestrianMovementFeatures:
    """
    A class for processing pedestrian scenarios to extract movement-related features
    such as speed, distance, and movement status.
    """

    def __init__(self, frames_per_second=5):
        """
        Initialize the PedestrianMovementFeatures class.

        Args:
            frames_per_second (int): Frame rate for calculating speed and movement status.
        """
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)
        self.frames_per_second = frames_per_second
        self.utils = PedestrianDataUtils()

    def extract_features_from_scenario(self, scenario_directory):
        """
        Process a single pedestrian scenario to extract features for each pedestrian.

        Args:
            scenario_directory (str): Directory containing odometry and label files for a scenario.

        Returns:
            dict: Dictionary of features for each pedestrian in each frame of the scenario.
        """
        try:
            self.logger.info(f"Extracting features from scenario: {scenario_directory}")

            # Initialize tracking variables
            previous_positions = {}
            pending_corrections = {}
            previous_ego_position = None
            scenario_features = {}

            odometry_files, label_files = self._get_files(scenario_directory)

            for odometry_file, label_file in zip(odometry_files, label_files):
                frame_id, ego_position, pedestrian_data = self._parse_frame_data(
                    scenario_directory, odometry_file, label_file
                )
                frame_features = self._extract_features_for_frame(
                    frame_id,
                    pedestrian_data,
                    previous_positions,
                    pending_corrections,
                    previous_ego_position,
                    ego_position,
                )
                scenario_features[frame_id] = frame_features
                previous_ego_position = ego_position

            self.logger.info(f"Feature extraction completed for scenario: {scenario_directory}")
            return scenario_features

        except Exception as e:
            self.logger.error(f"Error extracting features from scenario: {e}")
            raise e

    def process_all_pedestrian_scenarios(self, dataset_directory, output_directory):
        """
        Process all scenarios in the dataset and save extracted pedestrian features.

        Args:
            dataset_directory (str): Directory containing scenario subdirectories.
            output_directory (str): Directory to save JSON files with extracted features.
        """
        try:
            self.logger.info(f"Processing all pedestrian scenarios in: {dataset_directory}")
            os.makedirs(output_directory, exist_ok=True)

            for scenario in os.listdir(dataset_directory):
                scenario_path = os.path.join(dataset_directory, scenario)
                if not os.path.isdir(scenario_path):
                    continue

                self.logger.info(f"Processing scenario: {scenario}")
                scenario_features = self.extract_features_from_scenario(scenario_path)

                self._save_features_to_file(scenario, scenario_features, output_directory)

        except Exception as e:
            self.logger.error(f"Error processing pedestrian scenarios: {e}")
            raise e

    def _get_files(self, scenario_directory):
        """
        Retrieve and sort odometry and label files from a scenario directory.

        Args:
            scenario_directory (str): Path to the scenario directory.

        Returns:
            tuple: Lists of odometry files and label files.
        """
        frame_files = sorted(os.listdir(scenario_directory))
        odometry_files = [f for f in frame_files if f.startswith("odom")]
        label_files = [f for f in frame_files if f.startswith("label3d")]
        return odometry_files, label_files

    def _parse_frame_data(self, scenario_directory, odometry_file, label_file):
        """
        Parse data for a single frame, including odometry and pedestrian labels.

        Args:
            scenario_directory (str): Path to the scenario directory.
            odometry_file (str): Name of the odometry file.
            label_file (str): Name of the label file.

        Returns:
            tuple: Frame ID, ego vehicle position, and pedestrian data.
        """
        frame_id = int(odometry_file.split("_")[1].split(".")[0])
        odometry_path = os.path.join(scenario_directory, odometry_file)
        label_path = os.path.join(scenario_directory, label_file)

        ego_position = self.utils.parse_odometry(odometry_path)
        pedestrian_data = self.utils.parse_pedestrian_labels(label_path)

        return frame_id, ego_position, pedestrian_data

    def _extract_features_for_frame(
            self, frame_id, pedestrian_data, previous_positions, pending_corrections, previous_ego_position,
            ego_position
    ):
        """
        Extract features for all pedestrians in a single frame.

        Args:
            frame_id (int): Current frame ID.
            pedestrian_data (dict): Data for pedestrians in the current frame.
            previous_positions (dict): Pedestrian positions from the previous frame.
            pending_corrections (dict): Corrections pending for new pedestrians.
            previous_ego_position (dict): Ego vehicle position in the previous frame.
            ego_position (dict): Ego vehicle position in the current frame.

        Returns:
            dict: Features for each pedestrian in the current frame.
        """
        frame_features = {}
        for pedestrian_id, pedestrian_position in pedestrian_data.items():
            ped_x, ped_y = pedestrian_position["x"], pedestrian_position["y"]

            if ped_x < 0:  # Skip pedestrians behind the ego vehicle
                continue

            speed, distance, movement_status = self._process_pedestrian(
                pedestrian_id,
                (ped_x, ped_y),
                previous_positions,
                pending_corrections,
                previous_ego_position,
                ego_position,
                frame_id,
            )

            frame_features[pedestrian_id] = {
                "speed": speed,
                "distance": distance,
                "movement_status": movement_status,
            }

        return frame_features

    def _process_pedestrian(
            self, pedestrian_id, position, previous_positions, pending_corrections, previous_ego_position, ego_position,
            frame_id
    ):
        """
        Process a single pedestrian to compute speed, distance, and movement status.

        Args:
            pedestrian_id (str): Unique ID of the pedestrian.
            position (tuple): Current (x, y) position of the pedestrian.
            previous_positions (dict): Positions of pedestrians from the previous frame.
            pending_corrections (dict): Corrections pending for new pedestrians.
            previous_ego_position (dict): Ego vehicle position in the previous frame.
            ego_position (dict): Ego vehicle position in the current frame.
            frame_id (int): Current frame ID.

        Returns:
            tuple: Speed, distance, and movement status of the pedestrian.
        """
        if pedestrian_id in previous_positions:
            previous_position = previous_positions[pedestrian_id]
            speed, distance, movement_status = self.utils.calculate_pedestrian_metrics(
                previous_position, position, previous_ego_position, ego_position, self.frames_per_second
            )

            if pedestrian_id in pending_corrections:
                pending_frame_id = pending_corrections.pop(pedestrian_id)
                del previous_positions[pending_frame_id][pedestrian_id]
        else:
            speed, distance = 0, math.sqrt(position[0] ** 2 + position[1] ** 2)
            movement_status = "Unknown"
            pending_corrections[pedestrian_id] = frame_id

        previous_positions[pedestrian_id] = position
        return speed, distance, movement_status

    def _save_features_to_file(self, scenario, scenario_features, output_directory):
        """
        Save extracted features to a JSON file.

        Args:
            scenario (str): Scenario name.
            scenario_features (dict): Extracted features for the scenario.
            output_directory (str): Directory to save the JSON file.
        """
        output_path = os.path.join(output_directory, f"{scenario}_features.json")
        with open(output_path, "w") as json_file:
            json.dump(scenario_features, json_file, indent=4)
        self.logger.info(f"Features for scenario {scenario} saved to: {output_path}")
