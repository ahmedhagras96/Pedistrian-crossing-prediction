import torch
import json
import os
from attention_vector.Pedestrian_PC_Attention import PointNetFeatureExtractor
from pathlib import Path
import open3d as o3d
import numpy as np

# Save Features by Scenario
def save_features_by_scenario(features, output_directory):
    """
    Save extracted features to individual JSON files for each scenario.
    Args:
        features (dict): Dictionary of pedestrian IDs (file names) and their extracted features.
        output_directory (str): Path to the directory to save the scenario JSON files.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Organize features by scenario
    scenario_dict = {}
    for file_name, feature_vector in features.items():
        # Extract scenario identifier from file_name
        base_name = os.path.basename(file_name)
        scenario_id = base_name.split('_')[0]  # First part of the file name before '_'
        frame_number = base_name.split('_')[1]  # Second part of the file name after '_'

        # Initialize scenario entry if not already present
        if scenario_id not in scenario_dict:
            scenario_dict[scenario_id] = {}

        # Store the frame's features
        scenario_dict[scenario_id][f"frame_{frame_number}"] = feature_vector.tolist()

    # Save each scenario's data into separate JSON files
    for scenario_id, frames_data in scenario_dict.items():
        output_file = os.path.join(output_directory, f'scenario_{scenario_id}.json')
        with open(output_file, 'w') as json_file:
            json.dump(frames_data, json_file, indent=4)
        print(f"Scenario {scenario_id} features saved to {output_file}")

# Load Point Cloud
def load_point_cloud(file_path):
    """
    Load point cloud data from a .ply file.
    Args:
        file_path (str): Path to the .ply file.
    Returns:
        torch.Tensor: Tensor of shape [num_points, 3].
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)  # Convert to numpy array
    return torch.tensor(points, dtype=torch.float32)

def main():
    # Define paths
    ply_folder = "path_to_ply_folder"  
    output_directory = "path_to_output_directory"  

    # Initialize PointNetFeatureExtractor
    model = PointNetFeatureExtractor(input_dim=3, output_dim=64)
    model.eval()  # Set model to evaluation mode
    print(sum(p.numel() for p in model.parameters()), "parameters")

    # Dictionary to store features
    features = {}

    # Process each .ply file in the folder
    for ply_file in Path(ply_folder).glob("*.ply"):
        points = load_point_cloud(str(ply_file))
        points = points.unsqueeze(0)  # Add batch dimension, shape becomes [1, num_points, 3]

        feature = model(points).squeeze(0)  # Remove batch dimension, shape becomes [output_dim]

        # Store features with file name as the key
        features[str(ply_file)] = feature.cpu().numpy()

    # Save features organized by scenario
    #save_features_by_scenario(features, output_directory)

if __name__ == '__main__':
    main()
