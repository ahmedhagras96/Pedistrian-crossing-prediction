from torch.utils.data import DataLoader

from modules.avatar.pipeline.avatar_feature_extractor import AvatarFeatureExtractorPipeline
from modules.datasets.pedestrian_point_cloud_dataset import PedestrianPointCloudDataset
from modules.utilities.logger import LoggerUtils
from modules.utilities.scenario_manager import ScenarioManager


class PointCloudPipeline:
    """
    A pipeline for processing point clouds, extracting features, and saving results.

    Methods:
        run_pipeline():
            Execute the complete pipeline workflow.
    """

    def __init__(self, ply_folder, output_dir, batch_size, model, processor):
        """
        Initialize the PointCloudPipeline.

        Args:
            ply_folder (str): Path to the folder containing .ply files.
            output_dir (str): Directory to save scenario-based JSON files.
            batch_size (int): Batch size for feature extraction.
            model (PointNetFeatureExtractor): The PointNet model for feature extraction.
            processor (PointCloudProcessor): Processor for loading and normalizing point clouds.
        """
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

        self.ply_folder = ply_folder
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.processor = processor

        # Initialize components
        self.dataset = PedestrianPointCloudDataset(ply_folder, processor)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.dataset.collate_fn
        )
        self.feature_extractor = AvatarFeatureExtractorPipeline(model, batch_size)
        self.scenario_manager = ScenarioManager()

        self.logger.info("PointCloudPipeline initialized")

    def run_pipeline(self):
        """
        Execute the complete pipeline workflow:
            1. Extract features from point clouds.
            2. Group features by scenario.
            3. Save grouped features into JSON files.
        """
        self.logger.info("Starting the point cloud pipeline workflow")

        # Step 1: Extract features
        features = self.feature_extractor.extract_features(self.dataloader)

        # Step 2: Group features by scenario
        grouped_features = self.scenario_manager.group_by_scenario(features)

        # Step 3: Save grouped features to JSON files
        self.scenario_manager.save_to_json(grouped_features, self.output_dir)

        self.logger.info("Point cloud pipeline workflow completed successfully")
