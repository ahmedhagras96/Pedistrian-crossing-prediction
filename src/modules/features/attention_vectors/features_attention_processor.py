import json
import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from modules.config.logger import Logger
from modules.features.attention_vectors.multi_head_attention import MultiHeadAttention


class FeaturesAttentionProcessor:
    """
    A class to handle loading, preprocessing features from JSON, applying attention, 
    and saving the results.
    """
    logger = Logger.get_logger("FeaturesAttentionProcessor")

    @staticmethod
    def save_attention_results(output: torch.Tensor, attention_weights: torch.Tensor, output_file: str) -> None:
        """
        Save weighted output and attention weights to a JSON file.

        Args:
            output (torch.Tensor): The weighted output tensor.
            attention_weights (torch.Tensor): The attention weights tensor.
            output_file (str): Path to the output JSON file.

        Returns:
            None
        """
        output_data = {
            "weighted_output": output.detach().cpu().numpy().tolist(),
            "attention_weights": attention_weights.detach().cpu().numpy().tolist()
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        FeaturesAttentionProcessor.logger.info(f"Attention results saved to {output_file}")

    @staticmethod
    def load_features_from_json(json_file: str) -> torch.Tensor:
        """
        Load and preprocess features from a JSON file, scaling numerical values and converting to a tensor.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            torch.Tensor: Preprocessed features with a batch dimension.
        """
        FeaturesAttentionProcessor.logger.info(f"Loading features from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)

        features = []
        all_speeds = []
        all_distances = []

        # Collect speeds and distances for scaling
        for pedestrians in data.values():
            for ped_features in pedestrians.values():
                all_speeds.append(ped_features.get("speed", 0.0))
                all_distances.append(ped_features.get("distance", 0.0))

        speed_scaler = MinMaxScaler().fit(np.array(all_speeds).reshape(-1, 1))
        distance_scaler = MinMaxScaler().fit(np.array(all_distances).reshape(-1, 1))

        # Process and scale features
        for pedestrians in data.values():
            for ped_features in pedestrians.values():
                group_status = ped_features.get("group_status", 0)
                walking_toward_vehicle = ped_features.get("walking_toward_vehicle", 0)
                speed = speed_scaler.transform([[ped_features.get("speed", 0.0)]])[0][0]
                distance = distance_scaler.transform([[ped_features.get("distance", 0.0)]])[0][0]
                movement_status = 1 if ped_features.get("movement_status", "Stopped") == "Moving" else 0

                features.append([group_status, walking_toward_vehicle, speed, distance, movement_status])

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        FeaturesAttentionProcessor.logger.info(f"Features loaded with shape: {features_tensor.shape}")
        return features_tensor

    @classmethod
    def extract_features_attentions(
            cls, input_folder: str, output_folder: str, input_dim: int, output_dim: int, num_heads: int
    ) -> None:
        """
        Apply Multi-Head Attention to preprocessed features for each scenario JSON file in the input folder
        and save the results to the output folder.

        Args:
            input_folder (str): Folder containing input JSON files.
            output_folder (str): Folder where attention results will be saved.
            input_dim (int): Input dimension for the attention layer.
            output_dim (int): Output dimension for the attention layer.
            num_heads (int): Number of attention heads.

        Returns:
            None
        """
        cls.logger.info("Starting extraction of 3D features attentions")
        os.makedirs(output_folder, exist_ok=True)

        attention_layer = MultiHeadAttention(input_dim=input_dim, output_dim=output_dim, num_heads=num_heads)

        for json_file in os.listdir(input_folder):
            if json_file.endswith('.json'):
                input_path = os.path.join(input_folder, json_file)
                output_path = os.path.join(output_folder, f"attention_{json_file}")

                features = cls.load_features_from_json(input_path)
                cls.logger.info(f"Processing {json_file} with features shape: {features.shape}")

                attention_output, attention_weights = attention_layer(features)
                cls.save_attention_results(attention_output, attention_weights, output_path)
                cls.logger.info(f"Saved attention results for {json_file} to {output_path}")
