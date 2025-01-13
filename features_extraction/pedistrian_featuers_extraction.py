import os
import json
import numpy as np
# import torch
from sklearn.cluster import DBSCAN
# from distance_speed_move_stat_3D import get_speed_dist_ms_scenario
from .utils import *

def get_speed_dist_ms_scenario(scenario_path,fps):
    """
    Process a single scenario to extract features for each pedestrian,
    including corrections for initial "Unknown" movement status.
    """
    frame_files = sorted(os.listdir(scenario_path))
    odom_files = [f for f in frame_files if f.startswith("odom")]
    label_files = [f for f in frame_files if f.startswith("label3d")]

    # Track data
    previous_positions = {}
    pending_corrections = {}
    perv_ego_position = None

    scenario_features = {}

    for odom_file, label_file in zip(odom_files, label_files):
        cleaned_name = odom_file.split("_")[1].split(".")[0].split()[0]
        frame_id = int(cleaned_name)

        odom_path = os.path.join(scenario_path, odom_file)
        label_path = os.path.join(scenario_path, label_file)

        # Parse current odometry and labels
        ego_position = parse_odometry(odom_path)
        pedestrians = parse_labels(label_path)

        # Collect features for this frame
        frame_features = {}

        for ped_id, ped_data in pedestrians.items():
            ped_x, ped_y = ped_data["x"], ped_data["y"]

            # Ignore pedestrians behind the ego vehicle
            if ped_x < 0:
                continue

            if ped_id in previous_positions:
                # Calculate speed, distance, and movement status
                prev_position = previous_positions[ped_id]
                speed, distance, movement_status = calculate_speed_distance_movement(
                    prev_position, (ped_x, ped_y),perv_ego_position,ego_position,fps
                )

                # Remove pending corrections for this pedestrian if applicable
                if ped_id in pending_corrections:
                    pending_frame_id = pending_corrections[ped_id]
                    del scenario_features[pending_frame_id][ped_id]
                    del pending_corrections[ped_id]

            else:
                # Initialize for the first frame
                speed, distance = 0, math.sqrt(ped_x**2 + ped_y**2)
                movement_status = -1  #Unknown

                # Add this pedestrian to pending corrections
                pending_corrections[ped_id] = frame_id

            # Save current position for next frame
            previous_positions[ped_id] = (ped_x, ped_y)

            # Store features for this pedestrian
            frame_features[ped_id] = {
                "speed": speed,
                "distance": distance,
                "movement_status": movement_status
            }

        # Update scenario features
        scenario_features[frame_id] = frame_features

        # Update previous ego position
        perv_ego_position = ego_position

    return scenario_features

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

# def save_features(group_status, walking_status, output_file):
#     combined_features = {}
#     for frame_id in group_status.keys():
#         combined_features[frame_id] = {}
#         for ped_id, group_value in group_status[frame_id].items():
#             walking_value = walking_status.get(frame_id, {}).get(ped_id, 0)
#             combined_features[frame_id][ped_id] = {
#                 "group_status": group_value,
#                 "walking_toward_vehicle": walking_value
#             }

#     with open(output_file, 'w') as f:
#         json.dump(combined_features, f, indent=4)

# def save_features_per_pedestrian(scenario_id, group_status, walking_status, scenario_features, output_folder):
#     for frame_id in group_status.keys():
#         for ped_id, group_value in group_status[frame_id].items():
#             walking_value = walking_status.get(frame_id, {}).get(ped_id, 0)
            
#             pedestrian_features = {
#                 frame_id: {
#                     "group_status": group_value,
#                     "walking_toward_vehicle": walking_value
#                 }
#             }

#             if frame_id in scenario_features and ped_id in scenario_features[frame_id]:
#                 pedestrian_features[frame_id].update(scenario_features[frame_id][ped_id])

#             output_file = os.path.join(output_folder, f"{scenario_id}_{ped_id}_ped_{ped_id}.json")
            
#             if os.path.exists(output_file):
#                 with open(output_file, 'r') as f:
#                     existing_data = json.load(f)
#                     existing_data.update(pedestrian_features)
#             else:
#                 existing_data = pedestrian_features

#             with open(output_file, 'w') as f:
#                 json.dump(existing_data, f, indent=4)

def save_features_per_pedestrian(scenario_id, group_status, walking_status, scenario_features, output_folder,pid_avatars_dir):

    #get list of filtered pedistrians from the peistrian avatars directory
    avatar_ply_files = [f for f in os.listdir(pid_avatars_dir) if f.endswith('.ply')]
    filtered_peds = [os.path.splitext(f)[0] for f in avatar_ply_files]
    print("filtered_peds: ",filtered_peds)
    for frame_id in group_status.keys():
        for ped_id, group_value in group_status[frame_id].items():
            walking_value = walking_status.get(frame_id, {}).get(ped_id, 0)

            pedestrian_features = {
                "frame_id": frame_id,
                "group_status": group_value,
                "walking_toward_vehicle": walking_value
            }

            if frame_id in scenario_features and ped_id in scenario_features[frame_id]:
                pedestrian_features.update(scenario_features[frame_id][ped_id])

            zeros_frame_id = (4 - len(str(frame_id))) * "0"
            pedistrian_file_name = f"{scenario_id.split('_')[1]}_{zeros_frame_id}{frame_id}_ped_{ped_id}"

            if pedistrian_file_name not in filtered_peds:
                print(f"{pedistrian_file_name} not in filtered pedistrians")
                continue

            output_file = os.path.join(output_folder, f"{pedistrian_file_name}.json")

            with open(output_file, 'w') as f:
                json.dump(pedestrian_features, f, indent=4)

def extract_pedistrian_featuers(dataset_folder, output_folder,ped_avatar_dir):
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

        ##add distance and speed featuers
        scenario_path = os.path.join(dataset_folder, scenario_name)
        speed_distance_feas = get_speed_dist_ms_scenario(scenario_path, 5)

        # save_features(group_status, walking_status, output_file)
        save_features_per_pedestrian(scenario_name,group_status, walking_status,speed_distance_feas, output_folder,ped_avatar_dir)
        # print(f"Features saved for {scenario_name} to {output_file}")


# Example Usage
# dataset_folder = "LOKI"
# output_folder = "./processed_scenarios/output_features_Group & Walking"

# extract_pedistrian_featuers(dataset_folder, output_folder)
