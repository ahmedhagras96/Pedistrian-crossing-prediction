import torch
from torch.utils.data import DataLoader

from modules.utilities.logger import LoggerUtils


class AvatarFeatureExtractorPipeline:
    """
    A pipeline for batch-wise feature extraction from point cloud data using PointNet.

    Methods:
        extract_features(dataloader: DataLoader) -> Dict[str, np.ndarray]:
            Extract features from all batches using the provided DataLoader.
    """

    def __init__(self, model, batch_size):
        """
        Initialize the FeatureExtractorPipeline.

        Args:
            model (PointNetFeatureExtractor): An instance of the PointNet model.
            batch_size (int): Batch size for processing point clouds.
        """
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

        self.model = model
        self.batch_size = batch_size
        self.logger.info(f"FeatureExtractorPipeline initialized with batch_size={batch_size}")

    def extract_features(self, dataloader):
        """
        Extract features from all batches in the provided DataLoader.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader providing batches of point cloud data.

        Returns:
            dict: A dictionary mapping pedestrian IDs (file names) to their extracted features.
        """
        self.logger.info("Starting feature extraction process")
        features = {}
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (file_names, batch_points) in enumerate(dataloader):
                self.logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                batch_points_tensor = torch.stack(
                    [torch.tensor(points, dtype=torch.float32) for points in batch_points])
                batch_features = self.model.extract_features(batch_points_tensor)

                # Map features to pedestrian IDs (file names)
                for file_name, feature_vector in zip(file_names, batch_features):
                    features[file_name] = feature_vector.numpy()

        self.logger.info("Feature extraction completed")
        return features
