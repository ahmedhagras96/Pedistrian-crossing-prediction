from torch.utils.data import random_split, DataLoader

from modules.config.logger import LoggerUtils
from modules.model.datasets.pedestrian_point_cloud_dataset import PedestrianPointCloudDataset


# Import or define your dataset class here.
# from your_module import PedestrianPointCloudDataset

class DataLoaderUtils:
    """
    A utility class for dataset splitting and DataLoader creation with logging.
    """
    logger = LoggerUtils.get_logger("DataLoaderUtils")

    @staticmethod
    def random_split_dataset(dataset, train_set_percentage, val_set_percentage):
        """
        Randomly split the dataset into training, validation, and test sets.

        Args:
            dataset (Dataset): The dataset to split.
            train_set_percentage (float): Percentage of the dataset for training.
            val_set_percentage (float): Percentage of the dataset for validation.

        Returns:
            tuple: Training, validation, and test datasets.
        """
        logger = DataLoaderUtils.logger

        if train_set_percentage + val_set_percentage > 1:
            logger.error("Train and validation percentages sum exceeds 1.")
            raise ValueError("The sum of train_set_percentage and val_set_percentage should not exceed 1.")

        total_size = len(dataset)
        train_length = int(total_size * train_set_percentage)
        val_length = int(total_size * val_set_percentage)
        test_length = total_size - train_length - val_length

        logger.info(f"Splitting dataset of size {total_size} into "
                    f"train: {train_length}, val: {val_length}, test: {test_length}")

        return random_split(dataset, [train_length, val_length, test_length])

    @staticmethod
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
        logger = DataLoaderUtils.logger
        logger.info("Creating dataset and data loaders")

        # Initialize the dataset.
        dataset = PedestrianPointCloudDataset(pedestrian_dir, environment_dir, feature_dir, label_csv_path)

        # Split the dataset.
        train_set, val_set, test_set = DataLoaderUtils.random_split_dataset(
            dataset, train_set_percentage, val_set_percentage
        )

        # Create DataLoader objects.
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.collate_fn
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.collate_fn
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.collate_fn
        )

        logger.info("DataLoaders created successfully")
        return train_loader, val_loader, test_loader
