import os

from modules.avatar.pipeline.avatar_feature_extractor_pipeline import PointCloudPipeline
from modules.config.paths_loader import PathsLoader
from modules.models.point_net_feature_extractor import PointNetFeatureExtractor
from modules.utilities.point_cloud_utils import PointCloudUtils


def run_pipeline():
    processor = PointCloudUtils()
    model = PointNetFeatureExtractor(input_dim=3, output_dim=64)

    ply_folder = os.path.join(PathsLoader.get_folder_path(PathsLoader.Paths.PROCESSED_DATA), "saved_pedestrians")
    output_dir = os.path.join(PathsLoader.get_folder_path(PathsLoader.Paths.OUTPUT), "avatar_features")

    pipeline = PointCloudPipeline(
        ply_folder=ply_folder,
        output_dir=output_dir,
        batch_size=16,
        model=model,
        processor=processor
    )

    pipeline.run_pipeline()


if __name__ == "__main__":
    run_pipeline()
