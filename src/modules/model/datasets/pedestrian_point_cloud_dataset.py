import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from modules.utilities.point_cloud_utils import PointCloudUtils


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
        pedestrian_points = PointCloudUtils.load_and_normalize_ply(self.pedestrian_files[pedestrian_id])
        environment_points = PointCloudUtils.load_and_normalize_ply(self.environment_files[pedestrian_id])

        with open(self.feature_files[pedestrian_id], "r") as f:
            features = json.load(f)

        label = self.labels[pedestrian_id]
        pedestrian_features = torch.tensor(
            [
                features["group_status"],
                features["walking_toward_vehicle"],
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
