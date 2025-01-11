import json

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from modules.utilities.logger import LoggerUtils


class FeatureLoader:
    """
    A utility class for loading and preprocessing pedestrian features from JSON files.
    """

    def __init__(self):
        """Initialize the FeatureLoader and set up logging."""
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

    def load_features(self, json_file):
        """
        Load and preprocess features from a JSON file.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            torch.Tensor: Preprocessed feature tensor.
        """
        try:
            self.logger.info(f"Loading features from {json_file}")
            with open(json_file, "r") as f:
                data = json.load(f)

            features = []
            all_speeds = []
            all_distances = []

            for frame_id, pedestrians in data.items():
                for ped_id, ped_features in pedestrians.items():
                    all_speeds.append(ped_features.get("speed", 0.0))
                    all_distances.append(ped_features.get("distance", 0.0))

            speed_scaler = MinMaxScaler()
            distance_scaler = MinMaxScaler()
            speed_scaler.fit(np.array(all_speeds).reshape(-1, 1))
            distance_scaler.fit(np.array(all_distances).reshape(-1, 1))

            for frame_id, pedestrians in data.items():
                for ped_id, ped_features in pedestrians.items():
                    group_status = ped_features.get("group_status", 0)
                    walking_toward_vehicle = ped_features.get("walking_toward_vehicle", 0)
                    speed = speed_scaler.transform([[ped_features.get("speed", 0.0)]])[0][0]
                    distance = distance_scaler.transform([[ped_features.get("distance", 0.0)]])[0][0]
                    movement_status = 1 if ped_features.get("movement_status", "Stopped") == "Moving" else 0

                    features.append([group_status, walking_toward_vehicle, speed, distance, movement_status])

            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            self.logger.info(f"Loaded and processed features with shape: {features_tensor.shape}")
            return features_tensor

        except Exception as e:
            self.logger.error(f"Error loading features: {e}")
            raise e
