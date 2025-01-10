import os

import numpy as np
from torch.utils.data import Dataset

from modules.utilities.logger import LoggerUtils


class PedestrianPointCloudDataset(Dataset):
    """
    A custom dataset handler for managing pedestrian point cloud data.
    """

    def __init__(self, ply_folder, processor):
        """
        Initialize the dataset with pedestrian .ply files.

        Args:
            ply_folder (str): Path to the folder containing .ply files for pedestrians.
            processor (PointCloudProcessor): A point cloud processor instance for loading and normalizing data.
        """
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

        self.ply_folder = ply_folder
        self.processor = processor

        # Collect all .ply files in the folder
        self.ply_files = [os.path.join(ply_folder, f) for f in os.listdir(ply_folder) if f.endswith(".ply")]

        self.logger.info(f"Dataset initialized with {len(self.ply_files)} pedestrian .ply files in {ply_folder}")

    def __len__(self):
        """
        Returns:
            int: Number of pedestrian .ply files in the dataset.
        """
        return len(self.ply_files)

    def __getitem__(self, idx):
        """
        Load and normalize a pedestrian .ply file by index.

        Args:
            idx (int): Index of the pedestrian .ply file.

        Returns:
            tuple: (file_name, normalized_points)
        """
        file_name = self.ply_files[idx]
        self.logger.info(f"Loading pedestrian data from {file_name}")
        try:
            normalized_points = self.processor.process_ply(file_name)
            return file_name, normalized_points
        except Exception as e:
            self.logger.error(f"Error loading file {file_name}: {e}")
            raise e

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length point clouds in batches.

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
        padded_points = [
            np.pad(points, ((0, max_points - points.shape[0]), (0, 0)), mode="constant") for points in points_list
        ]

        return file_names, np.stack(padded_points)
