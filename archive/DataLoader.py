import os
import json
import pandas as pd
import numpy as np
import open3d as o3d

from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

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

class PedestrianPointCloudDataset(Dataset):
    def __init__(self, pedestrian_dir, environment_dir, feature_dir, label_csv):
        """
        Initialize dataset with pedestrian data, environment data, features, and labels.
        
        Args:
            pedestrian_dir (str): Path to the folder containing pedestrian .ply files.
            environment_dir (str): Path to the folder containing environment .ply files.
            feature_dir (str): Path to the folder containing feature .json files.
            label_csv (str): Path to the CSV file containing pedestrian IDs and labels.
        """
        # Load labels into a dictionary {pedestrian_id: label}
        self.labels = pd.read_csv(label_csv).set_index('id')['label'].to_dict()
        
        # Store paths to files in each directory
        self.pedestrian_files = {
            os.path.splitext(f)[0]: os.path.join(pedestrian_dir, f)
            for f in os.listdir(pedestrian_dir) if f.endswith(".ply")
        }
        self.environment_files = {
            os.path.splitext(f)[0]: os.path.join(environment_dir, f)
            for f in os.listdir(environment_dir) if f.endswith(".ply")
        }
        self.feature_files = {
            os.path.splitext(f)[0]: os.path.join(feature_dir, f)
            for f in os.listdir(feature_dir) if f.endswith(".json")
        }
        
        # Validate that all IDs have corresponding files
        self.pedestrian_ids = list(self.labels.keys())
        for pid in self.pedestrian_ids:
            if pid not in self.pedestrian_files or pid not in self.environment_files or pid not in self.feature_files:
                raise ValueError(f"Missing data for pedestrian ID: {pid}")

    def __len__(self):
        return len(self.pedestrian_ids)

    def __getitem__(self, idx):
        """
        Load data for a specific pedestrian.
        Args:
            idx (int): Index of the pedestrian in the dataset.
        Returns:
            dict: Dictionary containing pedestrian data, environment data, features, and label.
        """
        pedestrian_id = self.pedestrian_ids[idx]
        
        # Load pedestrian point cloud
        pedestrian_points = load_and_normalize_ply(self.pedestrian_files[pedestrian_id])
        
        # Load environment point cloud
        environment_points = load_and_normalize_ply(self.environment_files[pedestrian_id])
        
        # Load pre-extracted features
        with open(self.feature_files[pedestrian_id], 'r') as f:
            features = json.load(f)
        
        # Get label
        label = self.labels[pedestrian_id]
        
        return {
            "id": pedestrian_id,
            "pedestrian_points": pedestrian_points,
            "environment_points": environment_points,
            "features": features,
            "label": label
        }