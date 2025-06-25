import torch
import json
import sys

"""
if you want to run this file, run the following command: 
python -m attention_vector.point_cloud_attn_vector.output 
  
"""

# Ensure the script is run as a module
if __package__ is None or __package__ == "":
    print(
        "Error: This script must be run as a module. Use the following command:\n"
        "python -m attention_vector.point_cloud_attn_vector.output"
    )
    sys.exit(1)

from .attention_model import PointCloudAttentionModel

# Save Attention Results
def save_attention_results(output, attention_weights, output_file):
    """
    Save weighted output and attention weights to a JSON file.
    Args:
        output (torch.Tensor): Attention output tensor.
        attention_weights (torch.Tensor): Attention weights tensor.
        output_file (str): Path to save the output.
    """
    output_data = {
        "weighted_output": output.detach().cpu().numpy().tolist(),
        "attention_weights": attention_weights.detach().cpu().numpy().tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def main():
    # Define input points and parameters
    batch_size = 3
    num_points = 600
    points = torch.rand(batch_size, num_points, 3)

    model = PointCloudAttentionModel(embed_dim=128, kernel_size=3)
    print(sum(p.numel() for p in model.parameters()), "parameters")


    # Forward pass
    out, wei = model(points)

    print("Output shape:", out)
    print("Attention Weights shape:", wei.shape)
    
    # save_attention_results(out, wei, "LOKI/AttOut.json")
if __name__ == '__main__':
    main()