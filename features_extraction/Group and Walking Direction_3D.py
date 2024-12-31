import os
import json
import numpy as np
# import torch
from sklearn.cluster import DBSCAN


"""## Binary output of Group Feature"""
def load_label3d_positions(scenario_folder):
    label3d_files = sorted([f for f in os.listdir(scenario_folder) if f.startswith("label3d_") and f.endswith(".txt")])
    pedestrian_positions_frames = {}

    for file in label3d_files:
        frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))
        pedestrian_positions_frames[frame_id] = {}

        with open(os.path.join(scenario_folder, file), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if parts[0] == "Pedestrian":
                    ped_id = parts[1]
                    pos_x, pos_y, pos_z = map(float, parts[3:6])
                    pedestrian_positions_frames[frame_id][ped_id] = (pos_x, pos_y, pos_z)

    return pedestrian_positions_frames

def load_vehicle_positions(scenario_folder):
    odom_files = sorted([f for f in os.listdir(scenario_folder) if f.startswith("odom_") and f.endswith(".txt")])
    vehicle_positions = {}

    for file in odom_files:
        frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))

        with open(os.path.join(scenario_folder, file), 'r') as f:
            line = f.readline().strip()
            x, y, z, _, _, _ = map(float, line.split(','))
            vehicle_positions[frame_id] = (x, y, z)

    return vehicle_positions

def compute_group_status(pedestrian_positions_frames, proximity_threshold=5.0):
    group_status = {}

    for frame_id, pedestrian_positions in pedestrian_positions_frames.items():
        pedestrian_ids = list(pedestrian_positions.keys())
        coords = np.array(list(pedestrian_positions.values()))

        if len(coords) == 0:
            group_status[frame_id] = {}
            continue

        clustering = DBSCAN(eps=proximity_threshold, min_samples=2).fit(coords)
        labels = clustering.labels_

        group_status[frame_id] = {pedestrian_ids[i]: (1 if labels[i] != -1 else 0) for i in range(len(pedestrian_ids))}

    return group_status

def calculate_walking_toward_vehicle(pedestrian_positions_frames, vehicle_positions):
    walking_toward_vehicle = {}
    frame_ids = sorted(pedestrian_positions_frames.keys())

    for i in range(1, len(frame_ids)):
        frame_t_minus_1 = frame_ids[i - 1]
        frame_t = frame_ids[i]
        pedestrians_t_minus_1 = pedestrian_positions_frames[frame_t_minus_1]
        pedestrians_t = pedestrian_positions_frames[frame_t]
        vehicle_pos_t = vehicle_positions.get(frame_t, (0, 0, 0))

        walking_toward_vehicle[frame_t] = {}
        for ped_id, pos_t in pedestrians_t.items():
            if ped_id in pedestrians_t_minus_1:
                pos_t_minus_1 = pedestrians_t_minus_1[ped_id]

                movement_vector = np.array(pos_t) - np.array(pos_t_minus_1)
                toward_vehicle_vector = np.array(vehicle_pos_t) - np.array(pos_t)

                movement_norm = np.linalg.norm(movement_vector)
                toward_vehicle_norm = np.linalg.norm(toward_vehicle_vector)

                if movement_norm > 0 and toward_vehicle_norm > 0:
                    movement_unit = movement_vector / movement_norm
                    toward_vehicle_unit = toward_vehicle_vector / toward_vehicle_norm

                    cosine_similarity = np.dot(movement_unit, toward_vehicle_unit)
                    walking_toward_vehicle[frame_t][ped_id] = 1 if cosine_similarity > 0.7 else 0
                else:
                    walking_toward_vehicle[frame_t][ped_id] = 0

    return walking_toward_vehicle

def save_features(group_status, walking_status, output_file):
    combined_features = {}
    for frame_id in group_status.keys():
        combined_features[frame_id] = {}
        for ped_id, group_value in group_status[frame_id].items():
            walking_value = walking_status.get(frame_id, {}).get(ped_id, 0)
            combined_features[frame_id][ped_id] = {
                "group_status": group_value,
                "walking_toward_vehicle": walking_value
            }

    with open(output_file, 'w') as f:
        json.dump(combined_features, f, indent=4)

def process_all_scenarios(dataset_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for scenario_name in sorted(os.listdir(dataset_folder)):
        scenario_path = os.path.join(dataset_folder, scenario_name)

        if not os.path.isdir(scenario_path):
            continue

        print(f"Processing scenario: {scenario_name}...")
        output_file = os.path.join(output_folder, f"{scenario_name}_features.json")

        pedestrian_positions_frames = load_label3d_positions(scenario_path)
        vehicle_positions = load_vehicle_positions(scenario_path)
        group_status = compute_group_status(pedestrian_positions_frames)
        walking_status = calculate_walking_toward_vehicle(pedestrian_positions_frames, vehicle_positions)

        save_features(group_status, walking_status, output_file)
        print(f"Features saved for {scenario_name} to {output_file}")


# Example Usage
dataset_folder = "LOKI"
output_folder = "./processed_scenarios/output_features_Group & Walking"

process_all_scenarios(dataset_folder, output_folder)
