import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Multi_head_attention_layer import MultiHeadAttention

#merge all extracted 3D featuers (speed, distance, movmentstatus, Group status and walking direction)
def merge_features(group_walking_file, speed_distance_file, output_file):
    """
    Merge two JSON files for a scenario based on pedestrian IDs.
    """
    # Load JSON files
    with open(group_walking_file, "r") as gw_file:
        group_walking_data = json.load(gw_file)

    with open(speed_distance_file, "r") as sd_file:
        speed_distance_data = json.load(sd_file)

    # Initialize merged data
    merged_data = {}

    # Iterate through frames in both files
    for frame_id, ped_data_gw in group_walking_data.items():
        if frame_id in speed_distance_data:
            ped_data_sd = speed_distance_data[frame_id]
            merged_frame = {}

            # Match pedestrian IDs
            for ped_id, features_gw in ped_data_gw.items():
                if ped_id in ped_data_sd:
                    features_sd = ped_data_sd[ped_id]

                    # Concatenate features
                    merged_frame[ped_id] = {
                        "group_status": features_gw["group_status"],
                        "walking_toward_vehicle": features_gw["walking_toward_vehicle"],
                        "speed": features_sd["speed"],
                        "distance": features_sd["distance"],
                        "movement_status": features_sd["movement_status"]
                    }

            # Add merged frame data
            if merged_frame:
                merged_data[frame_id] = merged_frame

    # Save merged data to a new JSON file
    with open(output_file, "w") as out_file:
        json.dump(merged_data, out_file, indent=4)

    print(f"Merged features saved to {output_file}")

def merging_all_scenarios_featuers(group_walking_dir, speed_distance_dir, output_dir):
    """
    Process all scenarios by merging group/walking and speed/distance JSON files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each scenario
    for filename in os.listdir(group_walking_dir):
        if filename.endswith("_features.json"):
            # Derive corresponding file paths
            scenario_name = filename.replace("_features.json", "")
            group_walking_file = os.path.join(group_walking_dir, filename)
            speed_distance_file = os.path.join(speed_distance_dir, f"{scenario_name}_features.json")
            output_file = os.path.join(output_dir, f"{scenario_name}_merged_features.json")

            # Only merge if both files exist
            if os.path.exists(speed_distance_file):
                merge_features(group_walking_file, speed_distance_file, output_file)
            else:
                print(f"Missing speed/distance file for scenario: {scenario_name}")


# Define directories
group_walking_dir = "./processed_scenarios/output_features_Group & Walking"
speed_distance_dir = "./processed_scenarios/output_features_Speed & Distance"
output_dir = "./processed_scenarios/output_featuers_merged_jsons"

# Run the merging process
merging_all_scenarios_featuers(group_walking_dir, speed_distance_dir, output_dir)

# Save Attention Results
def save_attention_results(output, attention_weights, output_file):
    """
    Save weighted output and attention weights to a JSON file.
    """
    output_data = {
        "weighted_output": output.detach().cpu().numpy().tolist(),
        "attention_weights": attention_weights.detach().cpu().numpy().tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

# Load and Preprocess Features from JSON
def load_features_from_json(json_file):
    """
    Load and preprocess features from a JSON file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    features = []

    # Collect speed and distance values for scaling
    all_speeds = []
    all_distances = []
    for frame_id, pedestrians in data.items():
        for ped_id, ped_features in pedestrians.items():
            all_speeds.append(ped_features.get("speed", 0.0))
            all_distances.append(ped_features.get("distance", 0.0))

    # Fit scalers
    speed_scaler = MinMaxScaler()
    distance_scaler = MinMaxScaler()
    speed_scaler.fit(np.array(all_speeds).reshape(-1, 1))
    distance_scaler.fit(np.array(all_distances).reshape(-1, 1))

    # Process features
    for frame_id, pedestrians in data.items():
        for ped_id, ped_features in pedestrians.items():
            group_status = ped_features.get("group_status", 0)
            walking_toward_vehicle = ped_features.get("walking_toward_vehicle", 0)
            speed = speed_scaler.transform([[ped_features.get("speed", 0.0)]])[0][0]
            distance = distance_scaler.transform([[ped_features.get("distance", 0.0)]])[0][0]
            movement_status = 1 if ped_features.get("movement_status", "Stopped") == "Moving" else 0

            # Combine features
            features.append([group_status, walking_toward_vehicle, speed, distance, movement_status])

    # Convert to tensor and add batch dimension
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    return features_tensor

# Apply Attention to Each Scenario
def extrat_3D_featuers_attentions(input_folder, output_folder, input_dim, num_heads):
    """
    Apply Multi-Head Attention on preprocessed features for each scenario JSON file and save results.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    attention_layer = MultiHeadAttention(input_dim=input_dim, num_heads=num_heads)

    for json_file in os.listdir(input_folder):
        if json_file.endswith('.json'):
            input_path = os.path.join(input_folder, json_file)
            output_path = os.path.join(output_folder, f"attention_{json_file}")

            # Preprocess features
            features = load_features_from_json(input_path)
            print(f"Processing {json_file} with features shape: {features.shape}")

            # Apply attention
            attention_output, attention_weights = attention_layer(features)

            # Save results
            save_attention_results(attention_output, attention_weights, output_path)
            print(f"Saved attention results for {json_file} to {output_path}")


input_folder = "./processed_scenarios/output_featuers_merged_jsons"  # Folder containing JSON files for each scenario
output_folder = "./processed_scenarios/attention_results"  # Folder to save attention results
input_dim = 5  # Number of input features (group_status, walking_toward_vehicle, scaled_speed, scaled_distance, movement_status)
num_heads = 5  # Number of attention heads

extrat_3D_featuers_attentions(input_folder, output_folder, input_dim, num_heads)