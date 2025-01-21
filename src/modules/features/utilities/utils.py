import math
from typing import Dict, Tuple


def parse_odometry(file_path: str) -> Dict[str, float]:
    """
    Parse odometry data from a file.

    Args:
        file_path (str): Path to the odometry file.

    Returns:
        dict: Dictionary with keys 'x', 'y', and 'z' containing corresponding float values.
    """
    with open(file_path, "r") as f:
        values = list(map(float, f.readline().strip().split(",")))
        return {"x": values[0], "y": values[1], "z": values[2]}


def parse_labels(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Parse pedestrian labels from a file.

    Args:
        file_path (str): Path to the labels file.

    Returns:
        dict: A dictionary mapping pedestrian IDs to their 'x' and 'y' coordinates.
    """
    pedestrians = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if parts[0] == "Pedestrian":
                ped_id = parts[1]
                pos_x = float(parts[3])
                pos_y = float(parts[4])
                pedestrians[ped_id] = {"x": pos_x, "y": pos_y}
    return pedestrians


def calculate_speed_distance_movement(
        ped_coords_frame1: Tuple[float, float],
        ped_coords_frame2: Tuple[float, float],
        ego_odom_frame1: Dict[str, float],
        ego_odom_frame2: Dict[str, float],
        fps: float
) -> Tuple[float, float, int]:
    """
    Calculate pedestrian speed, distance from origin, and movement status between two frames.

    Args:
        ped_coords_frame1 (tuple): (x, y) coordinates of the pedestrian in frame 1.
        ped_coords_frame2 (tuple): (x, y) coordinates of the pedestrian in frame 2.
        ego_odom_frame1 (dict): Ego vehicle odometry with keys 'x', 'y', 'z' for frame 1.
        ego_odom_frame2 (dict): Ego vehicle odometry with keys 'x', 'y', 'z' for frame 2.
        fps (float): Frames per second.

    Returns:
        tuple:
            - speed (float): Pedestrian speed in units per second.
            - ped_distance (float): Distance of pedestrian from origin in frame 2.
            - movement_status (int): 1 if moving (speed >= 0.25), otherwise 0.
    """
    # Ego positions for frame 1 and frame 2
    ego_x1, ego_y1 = ego_odom_frame1["x"], ego_odom_frame1["y"]
    ego_x2, ego_y2 = ego_odom_frame2["x"], ego_odom_frame2["y"]

    # Pedestrian positions for frame 1 and frame 2
    ped_x1, ped_y1 = ped_coords_frame1
    ped_x2, ped_y2 = ped_coords_frame2

    # Calculate distance of pedestrian from origin in frame 2
    ped_distance = math.sqrt(ped_x2 ** 2 + ped_y2 ** 2)

    # Compute pedestrian relative position with respect to ego in frame 2
    relative_x2 = ped_x2 - ego_x2
    relative_y2 = ped_y2 - ego_y2

    # Difference in ego positions between frames
    diff_x_ego = ego_x2 - ego_x1
    diff_y_ego = ego_y2 - ego_y1

    # Adjust pedestrian position in frame 2 to frame 1 coordinate system
    adjusted_x2 = relative_x2 + ego_x1 + diff_x_ego
    adjusted_y2 = relative_y2 + ego_y1 + diff_y_ego

    # Compute distance traveled by pedestrian between frames 1 and adjusted frame 2
    distance = math.sqrt((adjusted_x2 - ped_x1) ** 2 + (adjusted_y2 - ped_y1) ** 2)

    # Calculate speed based on distance and time interval
    time_interval = 1 / fps
    speed = distance / time_interval

    # Determine movement status based on speed threshold
    movement_status = 0 if speed < 0.25 else 1

    return speed, ped_distance, movement_status
