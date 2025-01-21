import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.features.avatar.pedestrian_points_features.pedestrian_feature_extractor import PedestrianPointNetFeatureExtractor
from modules.features.avatar.pedestrian_points_features.utilities.pedestrian_pointcloud_dataset import PedestrianPointCloudDataset
from modules.config.logger import Logger


class FeatureBatchExtractor:
    """
    A utility class for extracting features in batches using a PointNet model.
    """

    _logger = Logger.get_logger("FeatureBatchExtractor")

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """
        Custom collate function to handle variable number of points per pedestrian.

        Args:
            batch (list): A list of tuples (file_name, points).

        Returns:
            tuple: A tuple containing:
                   file_names (list): List of file names.
                   padded_points (list): List of padded points arrays.
        """
        file_names = [item[0] for item in batch]
        points_list = [item[1] for item in batch]

        # Find the maximum number of points in the batch
        max_points = max(points.shape[0] for points in points_list)

        # Pad all points to the same size
        padded_points = [
            np.pad(points, ((0, max_points - points.shape[0]), (0, 0)), mode='constant')
            for points in points_list
        ]

        return file_names, padded_points

    @staticmethod
    def extract_features_in_batches(
            ply_folder: str,
            model: PedestrianPointNetFeatureExtractor,
            batch_size: int = 16
    ) -> dict:
        """
        Extract features from pedestrian .ply files using PointNet in batches.

        Args:
            ply_folder (str): Path to the folder containing .ply files.
            model (PointNetFeatureExtractor): A trained PointNet model.
            batch_size (int): Number of pedestrians per batch.

        Returns:
            dict: A dictionary where keys are file names and values are the extracted features.
        """
        FeatureBatchExtractor._logger.info(
            f"Extracting features from folder: {ply_folder} with batch_size={batch_size}"
        )

        # Create dataset and dataloader
        dataset = PedestrianPointCloudDataset(ply_folder)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=FeatureBatchExtractor.collate_fn
        )

        features = {}
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                file_names, batch_points = batch
                batch_points_tensor = torch.stack([
                    torch.tensor(points, dtype=torch.float32) for points in batch_points
                ])  # Shape: [batch_size, N, 3]

                FeatureBatchExtractor._logger.debug(
                    f"Processing batch with size: {batch_points_tensor.shape}"
                )

                features_tensor = model(batch_points_tensor)  # Shape: [batch_size, output_dim]

                # Map features to pedestrian file names
                for file_name, feature_vector in zip(file_names, features_tensor):
                    features[file_name] = feature_vector.numpy()

        FeatureBatchExtractor._logger.debug("Feature extraction completed.")
        return features
