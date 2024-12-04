
import os
import numpy as np
from sklearn.cluster import DBSCAN



"""## Binary output of Group Feature"""


def load_label3d(file_path):
    """
    Load 3D labels for pedestrians from the label3d.txt file.

    Args:
        file_path (str): Path to the label3d.txt file.

    Returns:
        dict: Dictionary with pedestrian IDs as keys and 3D coordinates as values.
    """
    pedestrian_positions = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if parts[0] == 'Pedestrian':  # Check if the label indicates a pedestrian
                try:
                    ped_id = parts[1]  # Track ID
                    pos_x, pos_y, pos_z = map(float, parts[3:6])  # Extract 3D coordinates
                    pedestrian_positions[ped_id] = (pos_x, pos_y, pos_z)
                except ValueError as e:
                    print(f"Error parsing line: {line} -> {e}")
    return pedestrian_positions


def check_groups_in_3d(scenario_folder, proximity_threshold=5.0):
    """
    Check if each pedestrian is in a group in each frame based on proximity in 3D space.

    Args:
        scenario_folder (str): Path to the folder containing 3D label3d files.
        proximity_threshold (float): Maximum distance (in meters) to consider pedestrians as part of the same group.

    Returns:
        dict: Dictionary containing frame IDs and pedestrian group statuses {pedestrian_id: 1 or 0}.
    """
    # Load all label3d files, sorted by frame number
    label3d_files = sorted([f for f in os.listdir(scenario_folder) if f.startswith("label3d_") and f.endswith(".txt")])

    frames_with_group_status = {}

    for label3d_file in label3d_files:
        try:
            frame_id = int(''.join(filter(str.isdigit, label3d_file.split('_')[1].split('.')[0])))
        except ValueError:
            print(f"Invalid file name: {label3d_file}")
            continue

        label3d_path = os.path.join(scenario_folder, label3d_file)
        pedestrian_positions = load_label3d(label3d_path)

        if not pedestrian_positions:
            frames_with_group_status[frame_id] = {}
            continue

        # Extract pedestrian IDs and coordinates
        pedestrian_ids = list(pedestrian_positions.keys())
        coords = np.array(list(pedestrian_positions.values()))

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=proximity_threshold, min_samples=2).fit(coords)
        labels = clustering.labels_

        # Map labels to pedestrian IDs and assign binary status
        frame_status = {}
        for idx, ped_id in enumerate(pedestrian_ids):
            label = labels[idx]
            if label == -1:
                frame_status[ped_id] = 0  # Not in a group
            else:
                frame_status[ped_id] = 1  # In a group

        frames_with_group_status[frame_id] = frame_status

    return frames_with_group_status


def process_all_scenarios(dataset_folder, proximity_threshold=5.0):
    """
    Process all scenarios in the dataset folder and check pedestrian groups for each scenario.

    Args:
        dataset_folder (str): Path to the folder containing all scenario subdirectories.
        proximity_threshold (float): Maximum distance (in meters) to consider pedestrians as part of the same group.

    Returns:
        dict: Dictionary containing scenario-wise group statuses.
              {scenario_id: {frame_id: {pedestrian_id: 1 or 0}}}.
    """
    all_scenarios_group_status = {}

    # Iterate through all scenario subdirectories
    for scenario_name in sorted(os.listdir(dataset_folder)):
        scenario_path = os.path.join(dataset_folder, scenario_name)

        # Skip non-directory files
        if not os.path.isdir(scenario_path):
            continue

        print(f"Processing scenario: {scenario_name}...")
        scenario_group_status = check_groups_in_3d(scenario_path, proximity_threshold)
        all_scenarios_group_status[scenario_name] = scenario_group_status

    return all_scenarios_group_status


# Example
dataset_folder = "/content/drive/MyDrive/Loki_Dataset/Loki"
proximity_threshold = 5.0  # Maximum distance in meters
all_scenarios_group_status = process_all_scenarios(dataset_folder, proximity_threshold)

# Print results for all scenarios
for scenario_id, frames in all_scenarios_group_status.items():
    print(f"Scenario: {scenario_id}")
    for frame_id, status_dict in frames.items():
        print(f"  Frame {frame_id}:")
        for ped_id, status in status_dict.items():
            print(f"    Pedestrian {ped_id}: {'In a group' if status == 1 else 'Not in a group'}")




"""## Extract Walking Directions"""

import os
import numpy as np


def load_label3d_positions(scenario_folder):
    """
    Extract pedestrian positions from label3d_*.txt files.

    Args:
        scenario_folder (str): Path to the folder containing label3d_*.txt files.

    Returns:
        dict: Pedestrian positions for each frame {frame_id: {pedestrian_id: (x, y, z)}}.
    """
    label3d_files = sorted([f for f in os.listdir(scenario_folder) if f.startswith("label3d_") and f.endswith(".txt")])
    pedestrian_positions_frames = {}

    for file in label3d_files:
        frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))
        pedestrian_positions_frames[frame_id] = {}

        with open(os.path.join(scenario_folder, file), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if parts[0] == "Pedestrian":  # Ensure it's a pedestrian
                    ped_id = parts[1]
                    pos_x, pos_y, pos_z = map(float, parts[3:6])  # Extract position
                    pedestrian_positions_frames[frame_id][ped_id] = (pos_x, pos_y, pos_z)

    return pedestrian_positions_frames


def load_vehicle_positions(scenario_folder):
    """
    Extract vehicle positions from odom_*.txt files.

    Args:
        scenario_folder (str): Path to the folder containing odom_*.txt files.

    Returns:
        dict: Vehicle positions for each frame {frame_id: (x, y, z)}.
    """
    odom_files = sorted([f for f in os.listdir(scenario_folder) if f.startswith("odom_") and f.endswith(".txt")])
    vehicle_positions = {}

    for file in odom_files:
        frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))

        with open(os.path.join(scenario_folder, file), 'r') as f:
            line = f.readline().strip()
            x, y, z, _, _, _ = map(float, line.split(','))  # Extract position
            vehicle_positions[frame_id] = (x, y, z)

    return vehicle_positions


def calculate_walking_toward_vehicle(pedestrian_positions_frames, vehicle_positions):
    """
    Calculate if pedestrians are walking toward the vehicle based on their walking direction.

    Args:
        pedestrian_positions_frames (dict): Pedestrian positions for consecutive frames.
            {frame_id: {pedestrian_id: (x, y, z)}}.
        vehicle_positions (dict): Vehicle positions for each frame {frame_id: (x, y, z)}.

    Returns:
        dict: Walking direction relative to the vehicle for each pedestrian.
            {pedestrian_id: [{"toward_vehicle": True/False, "cosine_similarity": value}, ...]}.
    """
    walking_toward_vehicle = {}

    # Get sorted frame IDs for consecutive frame comparison
    frame_ids = sorted(pedestrian_positions_frames.keys())

    for i in range(1, len(frame_ids)):  # Start from the second frame
        frame_t_minus_1 = frame_ids[i - 1]
        frame_t = frame_ids[i]

        pedestrians_t_minus_1 = pedestrian_positions_frames[frame_t_minus_1]
        pedestrians_t = pedestrian_positions_frames[frame_t]

        vehicle_pos_t = vehicle_positions.get(frame_t, (0, 0, 0))  # Default vehicle position if not provided

        for ped_id, pos_t in pedestrians_t.items():
            if ped_id in pedestrians_t_minus_1:
                pos_t_minus_1 = pedestrians_t_minus_1[ped_id]

                # Calculate pedestrian's movement vector
                movement_vector = np.array(pos_t) - np.array(pos_t_minus_1)

                # Calculate vector toward the vehicle
                toward_vehicle_vector = np.array(vehicle_pos_t) - np.array(pos_t)

                # Normalize vectors
                movement_norm = np.linalg.norm(movement_vector)
                toward_vehicle_norm = np.linalg.norm(toward_vehicle_vector)

                if movement_norm > 0 and toward_vehicle_norm > 0:
                    movement_unit = movement_vector / movement_norm
                    toward_vehicle_unit = toward_vehicle_vector / toward_vehicle_norm

                    # Compute cosine similarity
                    cosine_similarity = np.dot(movement_unit, toward_vehicle_unit)

                    # Determine if walking toward the vehicle
                    is_toward_vehicle = cosine_similarity > 0.7  # Threshold for "walking toward"
                else:
                    cosine_similarity = 0
                    is_toward_vehicle = False

                # Store results
                if ped_id not in walking_toward_vehicle:
                    walking_toward_vehicle[ped_id] = []
                walking_toward_vehicle[ped_id].append({
                    "toward_vehicle": is_toward_vehicle,
                    "cosine_similarity": cosine_similarity,
                    "movement_vector": movement_vector.tolist(),
                    "toward_vehicle_vector": toward_vehicle_vector.tolist()
                })

    return walking_toward_vehicle


def process_all_scenarios(dataset_folder):
    """
    Process all scenarios in the dataset folder and calculate pedestrian walking behavior.

    Args:
        dataset_folder (str): Path to the folder containing all scenario subdirectories.

    Returns:
        dict: Results for all scenarios in the format:
              {scenario_id: {pedestrian_id: [{"toward_vehicle": True/False, "cosine_similarity": value}, ...]}}
    """
    all_results = {}

    for scenario_name in sorted(os.listdir(dataset_folder)):
        scenario_path = os.path.join(dataset_folder, scenario_name)

        # Skip non-directory files
        if not os.path.isdir(scenario_path):
            continue

        print(f"Processing scenario: {scenario_name}...")
        pedestrian_positions_frames = load_label3d_positions(scenario_path)
        vehicle_positions = load_vehicle_positions(scenario_path)

        scenario_results = calculate_walking_toward_vehicle(pedestrian_positions_frames, vehicle_positions)
        all_results[scenario_name] = scenario_results

    return all_results


# Example
dataset_folder = "/content/drive/MyDrive/Loki_Dataset/Loki"
all_scenario_results = process_all_scenarios(dataset_folder)

# Print results for all scenarios
for scenario_id, pedestrian_results in all_scenario_results.items():
    print(f"Scenario: {scenario_id}")
    for ped_id, movements in pedestrian_results.items():
        print(f"  Pedestrian {ped_id}:")
        for movement in movements:
            print(f"    Toward Vehicle: {movement['toward_vehicle']}, Cosine Similarity: {movement['cosine_similarity']:.2f}")