import json
from pathlib import Path

import torch

from modules.attention.pointcloud_attention.attention_model import PointCloudAttentionModel
from modules.config.logger import Logger
from modules.config.paths_loader import PATHS

"""
if you want to run this file, run the following command: 
python -m attention.pointcloud_attention.output 
  
"""


# Ensure the script is run as a module
# if __package__ is None or __package__ == "":
#     print(
#         "Error: This script must be run as a module. Use the following command:\n"
#         "python -m attention.pointcloud_attention.output"
#     )
#     sys.exit(1)


def save_attention_results(output: torch.Tensor, attention_weights: torch.Tensor, output_file: str) -> None:
    """
    Save attention outputs and weights to a JSON file.

    Args:
        output (torch.Tensor):
            The attention output tensor of shape [B, embed_dim].
        attention_weights (torch.Tensor):
            The attention weights tensor of shape [B, M, num_heads].
        output_file (str):
            Path to the JSON file where the outputs will be saved.
    """

    output_data = {
        "weighted_output": output.detach().cpu().numpy().tolist(),
        "attention_weights": attention_weights.detach().cpu().numpy().tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    logger.debug(f"Attention results successfully saved to {output_file}")


if __name__ == '__main__':
    """
    Generates random point clouds, passes them through the model,
    and optionally saves the outputs and attention weights to a file.
    """
    Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("pointcloud_attention.log"))
    logger = Logger.get_logger("PointCloudAttentionOutput")

    # Define batch size, number of points
    batch_size = 3
    num_points = 600
    logger.info(f"Generating random points with batch_size={batch_size}, num_points={num_points}")

    # Create random points
    points = torch.rand(batch_size, num_points, 3)

    # Initialize model
    # logger.debug("Initializing PointCloudAttentionModel with embed_dim=128, kernel_size=3, num_heads=4.")
    model = PointCloudAttentionModel(embed_dim=128, kernel_size=3, num_heads=4)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params} parameters.")

    # Forward pass
    # logger.debug("Performing forward pass.")
    out, wei = model(points)

    logger.info(f"Output shape: {out.shape}")
    logger.info(f"Attention Weights shape: {wei.shape}")

    # Optionally save results (uncomment to use)
    save_attention_results(out, wei, PATHS.POINT_CLOUD_ATTENTION_JSON_FILE)
