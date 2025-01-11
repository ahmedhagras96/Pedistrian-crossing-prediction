import os

import torch

from modules.attention.point_cloud_attention.point_cloud_attention_model import PointCloudAttentionModel
from modules.config.paths_loader import PathsLoader
from modules.utilities.file_utils import FileUtils
from modules.utilities.logger import LoggerUtils


def run_point_cloud_attention_pipeline():
    """
    Main function to test the PointCloudAttentionModel with random input data.
    """
    # Define parameters
    batch_size = 3
    num_points = 600
    embed_dim = 8
    output_file = os.path.join(PathsLoader.get_folder_path(PathsLoader.Paths.OUTPUT), "attention",
                               "point_cloud_attention.json")

    # Initialize logger
    logger = LoggerUtils.get_logger(__name__)

    # Generate random point cloud data
    points = torch.rand(batch_size, num_points, 3)
    logger.debug(f"Generated random input point cloud of shape {points.shape}")

    # Initialize the model
    model = PointCloudAttentionModel(embed_dim=embed_dim)
    logger.info(f"Initialized PointCloudAttentionModel with embed_dim={embed_dim}")

    # Forward pass through the model
    logger.info("Performing forward pass through the point cloud attention model")
    try:
        output, attention_weights = model(points)
        logger.info(
            f"Forward pass completed. Output shape: {output.shape}, Attention weights shape: {attention_weights.shape}")
    except Exception as e:
        logger.error(f"Error during forward pass: {e}")
        return

    # Save the results to a file
    logger.info(f"Saving attention results to {output_file}")

    output_data = {
        "weighted_output": output.detach().cpu().numpy().tolist(),
        "attention_weights": attention_weights.detach().cpu().numpy().tolist(),
    }
    try:
        FileUtils.save_json(output_data, output_file)
        logger.info(f"Attention results successfully saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save attention results: {e}")


if __name__ == "__main__":
    run_point_cloud_attention_pipeline()
