import torch
import json
from attention_vector.Pedestrian_PC_Attention import PointNetFeatureExtractor
from pathlib import Path
import open3d as o3d
import numpy as np

# Save PointNet Features
def save_pointnet_features(output, output_file):
    """
    Save extracted PointNet features to a JSON file.
    Args:
        output (torch.Tensor): Feature output tensor.
        output_file (str): Path to save the output.
    """
    output_data = {
        "features": output.detach().cpu().numpy().tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Features saved to {output_file}")

def load_point_cloud(file_path):
    """
    Load point cloud data from a .ply file.
    Args:
        file_path (str): Path to the .ply file.
    Returns:
        torch.Tensor: Tensor of shape [num_points, 3].
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)  # Convert to numpy array
    return torch.tensor(points, dtype=torch.float32)

def main():
    # Define paths
    ply_file = "path_to_ply_file.ply"  # Replace with the actual .ply file path
    output_file = "PointNet_Features.json"  # Replace with your desired output file path

    # Load point cloud data
    points = load_point_cloud(ply_file)
    points = points.unsqueeze(0)  # Add batch dimension, shape becomes [1, num_points, 3]

    # Initialize PointNetFeatureExtractor
    model = PointNetFeatureExtractor(input_dim=3, output_dim=64)
    print(sum(p.numel() for p in model.parameters()), "parameters")

    # Forward pass
    features = model(points)

    print("Features shape:", features.shape)
    
    # Save features to JSON
    #save_pointnet_features(features, output_file)

if __name__ == '__main__':
    main()

