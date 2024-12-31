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

def calculate_speed_distance_movement(prev_position, curr_position, ego_delta):
    """
    Calculate speed, distance, and movement status for a pedestrian relative to the ego vehicle.
    """
    ped_x, ped_y = curr_position
    prev_x, prev_y = prev_position

    # Distance calculation
    distance = math.sqrt(ped_x**2 + ped_y**2)

    # Speed calculation
    movement = math.sqrt((ped_x - prev_x)**2 + (ped_y - prev_y)**2)
    adjusted_speed = max(0, movement - ego_delta)  # Adjust for ego movement

    # Movement status
    movement_status = "Stopped" if adjusted_speed < 0.25 else "Moving"

    return adjusted_speed, distance, movement_status