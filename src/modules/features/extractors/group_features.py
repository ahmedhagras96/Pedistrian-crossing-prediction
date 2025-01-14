import numpy as np
from sklearn.cluster import DBSCAN

from modules.utilities.logger import LoggerUtils


class GroupFeatures:
    """
    A class for processing a group of pedestrian in scenarios to extract features
    """

    def __init__(self):
        """
        Initialize the GroupFeatures class.
        """
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

    def calculate_walking_toward_vehicle(self, pedestrian_positions_frames, vehicle_positions):
        """
        Determine if pedestrians are walking toward the vehicle.
    
        Args:
            pedestrian_positions_frames (dict): Frame-wise pedestrian positions.
            vehicle_positions (dict): Frame-wise vehicle positions.
    
        Returns:
            dict: Mapping of frame IDs to pedestrians walking toward the vehicle (1 or 0).
        """
        self.logger.info("Calculating walking toward vehicle status.")
        walking_toward_vehicle = {}
        frame_ids = sorted(pedestrian_positions_frames.keys())

        for i in range(1, len(frame_ids)):
            frame_t_minus_1 = frame_ids[i - 1]
            frame_t = frame_ids[i]
            pedestrians_t_minus_1 = pedestrian_positions_frames[frame_t_minus_1]
            pedestrians_t = pedestrian_positions_frames[frame_t]
            vehicle_pos_t = vehicle_positions.get(frame_t, (0, 0, 0))

            walking_toward_vehicle[frame_t] = {}
            for ped_id, pos_t in pedestrians_t.items():
                if ped_id in pedestrians_t_minus_1:
                    pos_t_minus_1 = pedestrians_t_minus_1[ped_id]
                    movement_vector = np.array(pos_t) - np.array(pos_t_minus_1)
                    toward_vehicle_vector = np.array(vehicle_pos_t) - np.array(pos_t)
                    movement_norm = np.linalg.norm(movement_vector)
                    toward_vehicle_norm = np.linalg.norm(toward_vehicle_vector)
                    if movement_norm > 0 and toward_vehicle_norm > 0:
                        movement_unit = movement_vector / movement_norm
                        toward_vehicle_unit = toward_vehicle_vector / toward_vehicle_norm
                        cosine_similarity = np.dot(movement_unit, toward_vehicle_unit)
                        walking_toward_vehicle[frame_t][ped_id] = 1 if cosine_similarity > 0.7 else 0
                    else:
                        walking_toward_vehicle[frame_t][ped_id] = 0
        return walking_toward_vehicle

    def compute_group_status(self, pedestrian_positions_frames, proximity_threshold=5.0):
        """
        Compute group status for each pedestrian in each frame using DBSCAN clustering.
    
        Args:
            pedestrian_positions_frames (dict): Mapping of frame IDs to pedestrian positions.
            proximity_threshold (float): Distance threshold to consider pedestrians as a group.
    
        Returns:
            dict: Mapping of frame IDs to pedestrian group status (1 for group, 0 for isolated).
        """
        self.logger.info("Computing group status with DBSCAN clustering.")
        group_status = {}
        for frame_id, pedestrian_positions in pedestrian_positions_frames.items():
            pedestrian_ids = list(pedestrian_positions.keys())
            coords = np.array(list(pedestrian_positions.values()))
            if len(coords) == 0:
                group_status[frame_id] = {}
                continue
            clustering = DBSCAN(eps=proximity_threshold, min_samples=2).fit(coords)
            labels = clustering.labels_
            group_status[frame_id] = {pedestrian_ids[i]: (1 if labels[i] != -1 else 0)
                                      for i in range(len(pedestrian_ids))}
        return group_status
