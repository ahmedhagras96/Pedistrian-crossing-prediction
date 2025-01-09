import json
import os

import torch

from modules.attention_vector.point_cloud_attention.point_cloud_attention_model import PointCloudAttentionModel
from modules.config.paths_loader import PathsLoader
from modules.utilities.logger import LoggerUtils

# Initialize logger
logger = LoggerUtils.get_logger(__name__)


def save_attention_results(output: torch.Tensor, attention_weights: torch.Tensor, output_file: str):
    """
    Save the attention model's output and attention weights to a JSON file.

    Args:
        output (torch.Tensor): The output tensor from the attention model of shape [B, embed_dim].
        attention_weights (torch.Tensor): The attention weights tensor of shape [B, M, num_heads].
        output_file (str): File path to save the results as a JSON file.
    """
    logger.info(f"Saving attention results to {output_file}")

    output_data = {
        "weighted_output": output.detach().cpu().numpy().tolist(),
        "attention_weights": attention_weights.detach().cpu().numpy().tolist(),
    }
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Attention results successfully saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save attention results: {e}")


def main():
    """
    Main function to test the PointCloudAttentionModel with random input data.
    """
    # Define parameters
    batch_size = 3
    num_points = 600
    embed_dim = 8
    output_file = os.path.join(PathsLoader.get_folder_path(PathsLoader.Paths.OUTPUT), "point_cloud_attention.json")

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
    save_attention_results(output, attention_weights, output_file)


if __name__ == "__main__":
    main()
