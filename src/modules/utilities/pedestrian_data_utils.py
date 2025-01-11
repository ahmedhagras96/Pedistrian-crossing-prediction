import math

from modules.utilities.base_utility import BaseUtility


class PedestrianDataUtils(BaseUtility):
    """
    A utility class for parsing odometry and label data and calculating pedestrian metrics.
    """

    def __init__(self):
        super().__init__()

    def parse_odometry(self, file_path):
        """
        Parse odometry data from a file.

        Args:
            file_path (str): Path to the odometry file.

        Returns:
            dict: Dictionary containing 'x', 'y', 'z' coordinates.
        """
        try:
            self.logger.info(f"Parsing odometry data from {file_path}")
            with open(file_path, "r") as f:
                values = list(map(float, f.readline().strip().split(",")))
                return {"x": values[0], "y": values[1], "z": values[2]}
        except Exception as e:
            self.logger.error(f"Error parsing odometry data from {file_path}: {e}")
            raise e

    def parse_pedestrian_labels(self, file_path):
        """
        Parse pedestrian label data from a file.

        Args:
            file_path (str): Path to the label file.

        Returns:
            dict: Dictionary containing pedestrian IDs as keys and their 'x' and 'y' positions.
        """
        try:
            self.logger.info(f"Parsing pedestrian labels from {file_path}")
            pedestrians = {}
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if parts[0] == "Pedestrian":
                        pedestrian_id = parts[1]
                        pos_x = float(parts[3])
                        pos_y = float(parts[4])
                        pedestrians[pedestrian_id] = {"x": pos_x, "y": pos_y}
            return pedestrians
        except Exception as e:
            self.logger.error(f"Error parsing pedestrian labels from {file_path}: {e}")
            raise e

    def calculate_pedestrian_metrics(self, frame1_coords, frame2_coords, ego_frame1, ego_frame2, fps):
        """
        Calculate speed, distance, and movement status of a pedestrian between two frames.

        Args:
            frame1_coords (tuple): (x, y) coordinates of the pedestrian in frame 1.
            frame2_coords (tuple): (x, y) coordinates of the pedestrian in frame 2.
            ego_frame1 (dict): Ego vehicle's 'x', 'y' coordinates in frame 1.
            ego_frame2 (dict): Ego vehicle's 'x', 'y' coordinates in frame 2.
            fps (int): Frames per second of the data.

        Returns:
            tuple: A tuple containing:
                - speed (float): Speed of the pedestrian (units per second).
                - distance (float): Distance from the ego vehicle.
                - movement_status (str): "Moving" or "Stopped".
        """
        try:
            self.logger.debug("Calculating pedestrian metrics")

            # Ego vehicle positions
            ego_x1, ego_y1 = ego_frame1['x'], ego_frame1['y']
            ego_x2, ego_y2 = ego_frame2['x'], ego_frame2['y']

            # Pedestrian positions
            ped_x1, ped_y1 = frame1_coords
            ped_x2, ped_y2 = frame2_coords

            # Distance from pedestrian to ego vehicle
            pedestrian_distance = math.sqrt(ped_x2 ** 2 + ped_y2 ** 2)

            # Compute relative position in frame 2
            relative_x2 = ped_x2 - ego_x2
            relative_y2 = ped_y2 - ego_y2

            # Adjust relative position to frame 1 coordinates
            adjusted_x2 = relative_x2 + ego_x1 + (ego_x2 - ego_x1)
            adjusted_y2 = relative_y2 + ego_y1 + (ego_y2 - ego_y1)

            # Compute distance between frames
            distance_between_frames = math.sqrt((adjusted_x2 - ped_x1) ** 2 + (adjusted_y2 - ped_y1) ** 2)

            # Calculate speed
            time_interval = 1 / fps
            speed = distance_between_frames / time_interval

            # Determine movement status
            movement_status = "Stopped" if speed < 0.25 else "Moving"

            self.logger.debug(f"Metrics calculated: Speed={speed}, Distance={pedestrian_distance}, "
                              f"Movement Status={movement_status}")
            return speed, pedestrian_distance, movement_status
        except Exception as e:
            self.logger.error(f"Error calculating pedestrian metrics: {e}")
            raise e
