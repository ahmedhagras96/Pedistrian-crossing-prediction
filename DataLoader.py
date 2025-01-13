import os
import json
import pandas as pd
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence  
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def load_and_normalize_ply(ply_file):
    """
    Load and normalize 3D points from a .ply file.
    Args:
        ply_file (str): Path to the .ply file.
    Returns:
        np.ndarray: Normalized 3D points of shape (N, 3).
    """

    ply_file = str(ply_file)

    point_cloud = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(point_cloud.points)  # Shape: (N, 3)
    scaler = MinMaxScaler()
    normalized_points = scaler.fit_transform(points)
    return normalized_points

class PedestrianPointCloudDataset(Dataset):
    def __init__(self, pedestrian_dir, environment_dir, feature_dir, label_csv):
        """
        Initialize dataset with pedestrian data, environment data, features, and labels.
        """
        self.filtered_df = pd.read_csv(label_csv)
        self.scenario_ids = self.filtered_df['scenario_id']
        self.frame_ids = self.filtered_df['frame_id']
        self.labels = {f"{scenario_id:03}_{frame_id:04}_ped_{p_id}": status 
            for scenario_id, frame_id, p_id, status in
            zip(self.scenario_ids, self.frame_ids, self.filtered_df['track_id'], self.filtered_df['intended_actions'])
            }

        # Store paths to files
        self.pedestrian_files = {
            f.split(".")[0]: Path(pedestrian_dir) / f
            for f in os.listdir(pedestrian_dir) if f.endswith(".ply")
        }
        self.environment_files = {
            f.split(".")[0]: Path(environment_dir) / f
            for f in os.listdir(environment_dir) if f.endswith(".ply")
        }
        self.feature_files = {
            f.split(".")[0]: Path(feature_dir) / f
            for f in os.listdir(feature_dir) if f.endswith(".json")
        }
        self.pedestrian_ids = list(self.labels.keys())

        # Check that all files exist
        pedestrian_ids_in_files = list(self.pedestrian_files.keys())
        for p_id in self.pedestrian_ids:
            if p_id not in pedestrian_ids_in_files:
                raise ValueError(f"Missing pedestrian point cloud file for pedestrian ID: {p_id}")
            if p_id not in self.environment_files:
                raise ValueError(f"Missing environment point cloud file for pedestrian ID: {p_id}")
            if p_id not in self.feature_files:
                raise ValueError(f"Missing feature file for pedestrian ID: {p_id}")

    def __len__(self):
        return len(self.pedestrian_ids)

    def __getitem__(self, idx):
        pedestrian_id = self.pedestrian_ids[idx]
        pedestrian_points = load_and_normalize_ply(self.pedestrian_files[pedestrian_id])
        environment_points = load_and_normalize_ply(self.environment_files[pedestrian_id])

        with open(self.feature_files[pedestrian_id], 'r') as f:
            features = json.load(f)

        label = self.labels[pedestrian_id]
        pedestrian_features = torch.tensor([
            features["frame_id"],
            features["group_status"],
            features["walking_toward_vehicle"],
            # features["movement_status"],
            # features["speed"],
            # features["distance"]
        ])

        return (
            pedestrian_id,
            pedestrian_points,
            environment_points,
            pedestrian_features,
            label
        )

    def collate_fn(self, batch):
        """
        Collate function to handle variable-sized data in the batch using pad_sequence.
        """
        _, avatar_points, reconstructed_environment, pedestrian_features, labels = zip(*batch)

        # Convert to tensors
        avatar_points = [torch.tensor(p, dtype=torch.float32) for p in avatar_points]
        reconstructed_environment = [torch.tensor(e, dtype=torch.float32) for e in reconstructed_environment]

        # Pad sequences using pad_sequence
        batched_avatar_points = pad_sequence(avatar_points, batch_first=True)
        batched_reconstructed_environment = pad_sequence(reconstructed_environment, batch_first=True)

        # Stack features and labels
        batched_pedestrian_features = torch.stack([feat for feat in pedestrian_features])
        target = torch.tensor(labels, dtype=torch.int64)

        return batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target

# Create DataLoader
script_path = Path(__file__).resolve().parent
loki_path = script_path / "LOKI"
label_csv = loki_path / "b_avatar_filtered_pedistrians.csv"
pedestrian_dir = loki_path / 'training_data' / 'pedistrian_avatars'
environment_dir = loki_path / 'training_data' / '3d_constructed'
feature_dir = loki_path / 'training_data' / 'pedistrian_featuers'
batch_size = 8

train_ds = PedestrianPointCloudDataset(pedestrian_dir, environment_dir, feature_dir, label_csv)
dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=train_ds.collate_fn)

# Debugging
for batch in dataloader:
    batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target = batch
    print("Batched Avatar Points Shape:", batched_avatar_points.shape)
    print("Batched Reconstructed Environment Shape:", batched_reconstructed_environment.shape)
    print("Batched Pedestrian Features Shape:", batched_pedestrian_features.shape)
    print("Target Labels Shape:", target.shape)
    break
