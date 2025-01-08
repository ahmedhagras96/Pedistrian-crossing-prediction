import math

def parse_odometry(file_path):
    with open(file_path, "r") as f:
        values = list(map(float, f.readline().strip().split(",")))
        return {"x": values[0], "y": values[1], "z": values[2]}

def parse_labels(file_path):
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

def calculate_speed_distance_movement(ped_coords_frame1, ped_coords_frame2,ego_odom_frame1, ego_odom_frame2, fps):
    """
    Calculate pedestrian speed between two frames.

    Parameters:
        ego_odom_frame1: dict with 'x', 'y' for frame 1.
        ego_odom_frame2: dict with 'x', 'y' for frame 2.
        ped_coords_frame1: tuple (x, y) for pedestrian in frame 1.
        ped_coords_frame2: tuple (x, y) for pedestrian in frame 2.
        fps: Frames per second (e.g., 3 or 5).

    Returns:
        speed: Pedestrian speed in units per second.
    """

    # Ego car positions
    ego_x1, ego_y1 = ego_odom_frame1['x'], ego_odom_frame1['y']
    ego_x2, ego_y2 = ego_odom_frame2['x'], ego_odom_frame2['y']

    
    # Pedestrian positions in frames 1 and 2
    ped_x1, ped_y1 = ped_coords_frame1
    ped_x2, ped_y2 = ped_coords_frame2

    # current_pedistrian Distance calculation
    ped_distance = math.sqrt(ped_x2**2 + ped_y2**2)

    # Compute pedestrian relative position in frame 2
    relative_x2 = ped_x2 - ego_x2
    relative_y2 = ped_y2 - ego_y2

    #difference in current and pervious ego postion
    diff_x_ego = ego_x2 - ego_x1
    diff_y_ego = ego_y2 - ego_y1

    # Transform relative position to frame 1 coordinate system
    adjusted_x2 = relative_x2 + ego_x1 + diff_x_ego
    adjusted_y2 = relative_y2 + ego_y1 + diff_y_ego

    # Compute distance between pedestrian positions
    distance = math.sqrt((adjusted_x2 - ped_x1)**2 + (adjusted_y2 - ped_y1)**2)
    
    # Calculate speed
    time_interval = 1 / fps
    speed = distance / time_interval

    # Movement status
    movement_status = "Stopped" if speed < 0.25 else "Moving"
    
    return speed,ped_distance,movement_status