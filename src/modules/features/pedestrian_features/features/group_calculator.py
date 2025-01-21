from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


class GroupCalculator:
    """
    Utility class to compute group status and walking-toward-vehicle status for pedestrians.
    """

    @staticmethod
    def compute_group_status(
            pedestrian_positions_frames: Dict[int, Dict[str, Tuple[float, float, float]]],
            proximity_threshold: float = 5.0
    ) -> Dict[int, Dict[str, int]]:
        """
        Compute group status for each pedestrian in each frame using DBSCAN clustering.

        Args:
            pedestrian_positions_frames (dict): Frame-wise pedestrian positions.
            proximity_threshold (float): Distance threshold to consider pedestrians as a group.

        Returns:
            dict: Frame-wise group status for each pedestrian (1 if in group, else 0).
        """
        group_status: Dict[int, Dict[str, int]] = {}
        for frame_id, pedestrian_positions in pedestrian_positions_frames.items():
            pedestrian_ids = list(pedestrian_positions.keys())
            coords = np.array(list(pedestrian_positions.values()))

            if len(coords) == 0:
                group_status[frame_id] = {}
                continue

            clustering = DBSCAN(eps=proximity_threshold, min_samples=2).fit(coords)
            labels = clustering.labels_
            group_status[frame_id] = {
                pedestrian_ids[i]: (1 if labels[i] != -1 else 0)
                for i in range(len(pedestrian_ids))
            }
        return group_status

    @staticmethod
    def calculate_walking_toward_vehicle(
            pedestrian_positions_frames: Dict[int, Dict[str, Tuple[float, float, float]]],
            vehicle_positions: Dict[int, Tuple[float, float, float]]
    ) -> Dict[int, Dict[str, int]]:
        """
        Determine if pedestrians are walking toward the vehicle in each frame.

        Args:
            pedestrian_positions_frames (dict): Frame-wise pedestrian positions.
            vehicle_positions (dict): Frame-wise vehicle positions.

        Returns:
            dict: Frame-wise walking-toward-vehicle status for each pedestrian (1 if true, else 0).
        """
        walking_toward_vehicle: Dict[int, Dict[str, int]] = {}
        frame_ids = sorted(pedestrian_positions_frames.keys())

        for i in range(1, len(frame_ids)):
            frame_prev = frame_ids[i - 1]
            frame_curr = frame_ids[i]
            pedestrians_prev = pedestrian_positions_frames[frame_prev]
            pedestrians_curr = pedestrian_positions_frames[frame_curr]
            vehicle_pos = vehicle_positions.get(frame_curr, (0, 0, 0))

            walking_toward_vehicle[frame_curr] = {}
            for ped_id, pos_curr in pedestrians_curr.items():
                if ped_id in pedestrians_prev:
                    pos_prev = pedestrians_prev[ped_id]
                    movement_vector = np.array(pos_curr) - np.array(pos_prev)
                    toward_vehicle_vector = np.array(vehicle_pos) - np.array(pos_curr)

                    movement_norm = np.linalg.norm(movement_vector)
                    toward_vehicle_norm = np.linalg.norm(toward_vehicle_vector)

                    if movement_norm > 0 and toward_vehicle_norm > 0:
                        movement_unit = movement_vector / movement_norm
                        toward_vehicle_unit = toward_vehicle_vector / toward_vehicle_norm
                        cosine_similarity = np.dot(movement_unit, toward_vehicle_unit)
                        walking_toward_vehicle[frame_curr][ped_id] = 1 if cosine_similarity > 0.7 else 0
                    else:
                        walking_toward_vehicle[frame_curr][ped_id] = 0
        return walking_toward_vehicle
