import os

from torch.utils.data import Dataset

from modules.features.avatar.pedestrian_points_features.utilities.ply_processor import PlyProcessor
from modules.config.logger import Logger


class PedestrianPointCloudDataset(Dataset):
    """
    A PyTorch Dataset for pedestrian point cloud files in .ply format.
    """

    _logger = Logger.get_logger("PedestrianPointCloudDataset")

    def __init__(self, ply_folder: str):
        """
        Initialize the dataset with pedestrian point cloud files.

        Args:
            ply_folder (str): Path to the folder containing .ply files.
        """
        super().__init__()
        self.ply_folder = ply_folder
        self.ply_files = [
            os.path.join(ply_folder, f)
            for f in os.listdir(ply_folder)
            if f.endswith(".ply")
        ]

        self._logger.info(
            f"Found {len(self.ply_files)} .ply files in folder: {ply_folder}"
        )

    def __len__(self) -> int:
        """
        Number of samples in the dataset.

        Returns:
            int: Total number of .ply files.
        """
        return len(self.ply_files)

    def __getitem__(self, idx: int) -> tuple:
        """
        Load and normalize a pedestrian point cloud.

        Args:
            idx (int): Index of the .ply file.

        Returns:
            tuple: (file_name, normalized_points)
        """
        file_name = self.ply_files[idx]
        normalized_points = PlyProcessor.load_and_normalize_ply(file_name)

        self._logger.debug(f"Loaded file: {file_name} with shape {normalized_points.shape}")
        return file_name, normalized_points
