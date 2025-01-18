import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import open3d as o3d
from sklearn.preprocessing import MinMaxScaler

# Load and Normalize Points from .ply
def load_and_normalize_ply(ply_file):
    """
    Load and normalize 3D points from a .ply file.
    Args:
        ply_file (str): Path to the .ply file.
    Returns:
        np.ndarray: Normalized 3D points of shape (N, 3).
    """
    # Load the .ply file using Open3D
    point_cloud = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(point_cloud.points)  # Shape: (N, 3)

    # Normalize points to [0, 1] using Min-Max scaling
    scaler = MinMaxScaler()
    normalized_points = scaler.fit_transform(points)

    return normalized_points

class PointNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super(PointNetFeatureExtractor, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): Tensor of shape [B, N, 3] (Batch, Num Points, Features)
        Returns:
            torch.Tensor: Tensor of shape [B, output_dim] (Aggregated Features for each pedestrian)
        """
        B, N, _ = points.shape
        x = self.mlp1(points)  # Shape: [B, N, 256]
        x = x.transpose(1, 2)  # Shape: [B, 256, N]
        x = self.global_pool(x).squeeze(-1)  # Shape: [B, 256]
        x = self.fc(x)  # Shape: [B, output_dim]
        return x

# Dataset for Batch Processing
class PedestrianPointCloudDataset(Dataset):
    def __init__(self, ply_folder):
        """
        Initialize dataset with pedestrian point cloud files.
        Args:
            ply_folder (str): Path to the folder containing .ply files for pedestrians.
        """
        self.ply_files = [os.path.join(ply_folder, f) for f in os.listdir(ply_folder) if f.endswith(".ply")]

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        """
        Load and normalize a pedestrian point cloud.
        Args:
            idx (int): Index of the pedestrian .ply file.
        Returns:
            tuple: (file_name, normalized_points)
        """
        file_name = self.ply_files[idx]
        points = load_and_normalize_ply(file_name)
        return file_name, points
    
# Load and Normalize Points from .ply
def load_and_normalize_ply(ply_file):
    """
    Load and normalize 3D points from a .ply file.
    Args:
        ply_file (str): Path to the .ply file.
    Returns:
        np.ndarray: Normalized 3D points of shape (N, 3).
    """
    # Load the .ply file using Open3D
    point_cloud = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(point_cloud.points)  # Shape: (N, 3)

    # Normalize points to [0, 1] using Min-Max scaling
    scaler = MinMaxScaler()
    normalized_points = scaler.fit_transform(points)

    return normalized_points

# Batch Processing for Feature Extraction
def extract_features_in_batches(ply_folder, model, batch_size=16):
    """
    Extract features from pedestrian .ply files using PointNet in batches.
    Args:
        ply_folder (str): Path to the folder containing .ply files.
        model (PointNetFeatureExtractor): Trained PointNet model.
        batch_size (int): Number of pedestrians per batch.
    Returns:
        dict: Dictionary of pedestrian IDs (file names) and their extracted features.
    """
    # Create dataset and dataloader
    dataset = PedestrianPointCloudDataset(ply_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Extract features
    features = {}
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            file_names, batch_points = batch
            batch_points_tensor = torch.stack([torch.tensor(points, dtype=torch.float32) for points in batch_points])
            features_tensor = model(batch_points_tensor)  # Shape: [batch_size, output_dim]

            # Map features to pedestrian IDs (file names)
            for file_name, feature_vector in zip(file_names, features_tensor):
                features[file_name] = feature_vector.numpy()

    return features

# Custom Collate Function for Dataloader
def collate_fn(batch):
    """
    Custom collate function to handle variable number of points per pedestrian.
    Args:
        batch (list): List of tuples (file_name, points).
    Returns:
        tuple: File names and padded points tensors.
    """
    file_names = [item[0] for item in batch]
    points_list = [item[1] for item in batch]

    # Find the maximum number of points in the batch
    max_points = max(points.shape[0] for points in points_list)

    # Pad all points to the same size
    padded_points = [np.pad(points, ((0, max_points - points.shape[0]), (0, 0)), mode='constant') for points in points_list]
    return file_names, padded_points


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
        scenario_id = base_name.split('_')[0]  # First three digits
        frame_number = base_name.split('_')[1]  # Frame number

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


# Updated batch feature extraction with saving by scenario
def extract_and_save_features_by_scenario(ply_folder, model, batch_size, output_directory):
    """
    Extract features from pedestrian .ply files using PointNet and save by scenario.
    Args:
        ply_folder (str): Path to the folder containing .ply files.
        model (PointNetFeatureExtractor): Trained PointNet model.
        batch_size (int): Number of pedestrians per batch.
        output_directory (str): Directory to save the scenario JSON files.
    """
    # Extract features
    features = extract_features_in_batches(ply_folder, model, batch_size)

    # Save features by scenario
    save_features_by_scenario(features, output_directory)

# Initialize the model
pointnet_model = PointNetFeatureExtractor(input_dim=3, output_dim=64)

# Path to the folder containing .ply files
ply_folder = "processed_scenarios\saved_pedestrians"

# Output JSON file path
scenario_output_directory = "processed_scenarios\extracted_pedistrian_avatar_featuers"

# Extract features and save by scenario
extract_and_save_features_by_scenario(ply_folder, pointnet_model, batch_size=2, output_directory=scenario_output_directory)