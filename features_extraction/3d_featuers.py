import math
import os

def read_odometry(file_path):
    """
    Reads odometry data from a file and extracts ego-car position (x, y).
    
    Args:
        file_path (str): Path to the odometry file.
        
    Returns:
        tuple: Ego-car position (x, y).
    """
    with open(file_path, 'r') as file:
        line = file.readline().strip().split(',')
        pos_x, pos_y = float(line[0]), float(line[1])
    return pos_x, pos_y

def read_pedestrian_position(file_path, track_id):
    """
    Reads pedestrian position from a label file based on track_id.
    
    Args:
        file_path (str): Path to the label3d file.
        track_id (str): Unique ID of the pedestrian.
        
    Returns:
        tuple: Pedestrian position (x, y), or None if track_id not found.
    """
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            if data[1] == track_id:  # Match track_id
                pos_x, pos_y = float(data[3]), float(data[4])
                return pos_x, pos_y
    return None

def calculate_2d_distance(pos1, pos2):
    """
    Calculate the 2D Euclidean distance between two points.
    
    Args:
        pos1 (tuple): First position (x, y).
        pos2 (tuple): Second position (x, y).
        
    Returns:
        float: 2D distance.
    """
    return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

def main(file_id, track_id, odometry_dir, label_dir):
    """
    Main function to compute the 2D distance between ego-car and a pedestrian.
    
    Args:
        file_id (str): File ID to match odometry and label files.
        track_id (str): Unique ID of the pedestrian.
        odometry_dir (str): Directory containing odometry files.
        label_dir (str): Directory containing label files.
        
    Returns:
        float: 2D distance between ego-car and the pedestrian.
    """
    odom_file = os.path.join(odometry_dir, f"odom_{file_id}.txt")
    label_file = os.path.join(label_dir, f"label3d_{file_id}.txt")
    
    # Read positions
    ego_car_pos = read_odometry(odom_file)
    pedestrian_pos = read_pedestrian_position(label_file, track_id)
    
    if pedestrian_pos is None:
        raise ValueError(f"Pedestrian with track_id '{track_id}' not found in {label_file}")
    
    # Calculate and return distance
    return calculate_2d_distance(ego_car_pos, pedestrian_pos)

# Example usage
if __name__ == "__main__":
    # File ID and track_id for testing
    file_id = "0002"
    track_id = "13f222fd-065c-4745-b441-44dd25566cbb"
    
    # Directories
    odometry_dir = "/path/to/odometry/files"
    label_dir = "/path/to/label/files"
    
    # Compute distance
    try:
        distance = main(file_id, track_id, odometry_dir, label_dir)
        print(f"Distance between ego-car and pedestrian (track_id={track_id}): {distance:.2f} meters")
    except ValueError as e:
        print(e)
