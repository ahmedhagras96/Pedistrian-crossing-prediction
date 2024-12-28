import torch
import json

from attention_model import PointCloudAttentionModel

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
    batch_size = 8
    num_points = 1000
    points = torch.rand(batch_size, num_points, 3)

    model = PointCloudAttentionModel(embed_dim=8)

    # Forward pass
    out, wei = model(points)

    print("Output shape:", out.shape)
    print("Attention Weights shape:", wei.shape)
    
    save_attention_results(out, wei, "LOKI/AttOut.json")


if __name__ == "__main__":
    main()