import os
import json
import pandas as pd
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def load_and_normalize_ply(ply_file):
    """
    Load and normalize 3D points from a .ply file.
    
    Args:
        ply_file (str or Path): Path to the .ply file.
    
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
    """
    A PyTorch Dataset class for loading pedestrian point cloud data, environment data, and features.
    """

    def __init__(self, pedestrian_dir, environment_dir, feature_dir, label_csv):
        """
        Initialize the dataset.

        Args:
            pedestrian_dir (str): Directory containing pedestrian .ply files.
            environment_dir (str): Directory containing environment .ply files.
            feature_dir (str): Directory containing feature .json files.
            label_csv (str): Path to the CSV file containing pedestrian IDs and labels.
        """
        self.filtered_df = pd.read_csv(label_csv)
        self.scenario_ids = self.filtered_df["scenario_id"]
        self.frame_ids = self.filtered_df["frame_id"]
        self.labels = {
            f"{scenario_id:03}_{frame_id:04}_ped_{p_id}": status
            for scenario_id, frame_id, p_id, status in zip(
                self.scenario_ids, self.frame_ids, self.filtered_df["track_id"], self.filtered_df["intended_actions"]
            )
        }

        # Store paths to files
        self.pedestrian_files = {
            f.split(".")[0]: Path(pedestrian_dir) / f for f in os.listdir(pedestrian_dir) if f.endswith(".ply")
        }
        self.environment_files = {
            f.split(".")[0]: Path(environment_dir) / f for f in os.listdir(environment_dir) if f.endswith(".ply")
        }
        self.feature_files = {
            f.split(".")[0]: Path(feature_dir) / f for f in os.listdir(feature_dir) if f.endswith(".json")
        }
        self.pedestrian_ids = list(self.labels.keys())

        # Validate that all required files exist
        self._validate_files()

        # Load all features to determine scaling parameters
        self.scaler = MinMaxScaler()
        self._compute_scaling_parameters()

    def _validate_files(self):
        """
        Validate that all required files exist for each pedestrian ID.
        """
        pedestrian_ids_in_files = list(self.pedestrian_files.keys())
        for p_id in self.pedestrian_ids:
            if p_id not in pedestrian_ids_in_files:
                raise ValueError(f"Missing pedestrian point cloud file for pedestrian ID: {p_id}")
            if p_id not in self.environment_files:
                raise ValueError(f"Missing environment point cloud file for pedestrian ID: {p_id}")
            if p_id not in self.feature_files:
                raise ValueError(f"Missing feature file for pedestrian ID: {p_id}")

    def _compute_scaling_parameters(self):
        """
        Compute the scaling parameters for the speed and distance features across the entire dataset.
        """
        features = []
        for pedestrian_id in self.pedestrian_ids:
            with open(self.feature_files[pedestrian_id], "r") as f:
                data = json.load(f)
                features.append([data["speed"], data["distance"]])

        features = np.array(features)
        self.scaler.fit(features)
    
    def __len__(self):
        return len(self.pedestrian_ids)

    def __getitem__(self, idx):
        """
        Get a single data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Contains pedestrian ID, pedestrian points, environment points, features, and label.
        """
        pedestrian_id = self.pedestrian_ids[idx]
        pedestrian_points = load_and_normalize_ply(self.pedestrian_files[pedestrian_id])
        environment_points = load_and_normalize_ply(self.environment_files[pedestrian_id])

        with open(self.feature_files[pedestrian_id], "r") as f:
            features = json.load(f)

        label = self.labels[pedestrian_id]

        # Normalize speed and distance
        speed, distance = self.scaler.transform(
            np.array([[features["speed"], features["distance"]]])
        )[0]
        movement_status = 1 if features["movement_status"] == 1 else 0  # Binary value: 0 or 1

        pedestrian_features = torch.tensor(
            [
                features["group_status"],
                features["walking_toward_vehicle"],
                speed,
                distance,
                movement_status,
            ],
            dtype=torch.float32,
        )

        return pedestrian_id, pedestrian_points, environment_points, pedestrian_features, label

    def collate_fn(self, batch):
        """
        Collate function to handle variable-sized data in the batch.

        Args:
            batch (list): List of samples from the dataset.

        Returns:
            tuple: Batched tensors for avatar points, environment points, features, and labels.
        """
        _, avatar_points, reconstructed_environment, pedestrian_features, labels = zip(*batch)

        # Convert to tensors
        avatar_points = [torch.tensor(p, dtype=torch.float32) for p in avatar_points]
        reconstructed_environment = [torch.tensor(e, dtype=torch.float32) for e in reconstructed_environment]

        # Pad sequences using pad_sequence
        batched_avatar_points = pad_sequence(avatar_points, batch_first=True)
        batched_reconstructed_environment = pad_sequence(reconstructed_environment, batch_first=True)

        # Stack features and labels
        batched_pedestrian_features = torch.stack(pedestrian_features)
        target = torch.tensor(labels, dtype=torch.int64)

        return batched_avatar_points, batched_reconstructed_environment, batched_pedestrian_features, target


def random_split_dataset(dataset, train_set_percentage, val_set_percentage):
    """
    Randomly split the dataset into training, validation, and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_set_percentage (float): Percentage of the dataset to use for training.
        val_set_percentage (float): Percentage of the dataset to use for validation.

    Returns:
        tuple: Training, validation, and test datasets.
    """
    if train_set_percentage + val_set_percentage > 1:
        raise ValueError("The sum of train_set_percentage and val_set_percentage should not exceed 1.")

    train_length = int(len(dataset) * train_set_percentage)
    val_length = int(len(dataset) * val_set_percentage)
    test_length = len(dataset) - train_length - val_length

    return random_split(dataset, [train_length, val_length, test_length])


def get_data_loaders(
    pedestrian_dir,
    environment_dir,
    feature_dir,
    label_csv_path,
    batch_size,
    train_set_percentage=0.7,
    val_set_percentage=0.2,
    shuffle=True,
    drop_last=True,
):
    """
    Create DataLoader objects for training, validation, and testing.

    Args:
        pedestrian_dir (str): Directory containing pedestrian .ply files.
        environment_dir (str): Directory containing environment .ply files.
        feature_dir (str): Directory containing feature .json files.
        label_csv_path (str): Path to the CSV file containing pedestrian IDs and labels.
        batch_size (int): Batch size for the DataLoader.
        train_set_percentage (float): Percentage of the dataset to use for training.
        val_set_percentage (float): Percentage of the dataset to use for validation.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
        tuple: Training, validation, and test DataLoader objects.
    """
    dataset = PedestrianPointCloudDataset(pedestrian_dir, environment_dir, feature_dir, label_csv_path)
    train_set, val_set, test_set = random_split_dataset(dataset, train_set_percentage, val_set_percentage)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.collate_fn
    )

    return train_loader, val_loader, test_loader