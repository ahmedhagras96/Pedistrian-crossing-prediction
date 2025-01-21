import os
import threading
from typing import Dict

import numpy as np
import open3d as o3d
import pandas as pd

from modules.config.logger import Logger


class PedestrianPCDManager:
    """
    Handles preparation and saving of pedestrian point clouds (PCDs).
    """

    def __init__(self, save_dir: str) -> None:
        """
        Initializes the manager for pedestrian PCDs.

        Args:
            save_dir (str): Directory where pedestrian PCDs will be saved.
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.logger.info(f"Created directory for saving pedestrian PCDs: {self.save_dir}")

    def prepare_pedestrian_pcds(
            self,
            scenario_id: str,
            frame_id: str,
            df_pedestrians_filtered: pd.DataFrame,
            pedestrian_pcds_filtered: list
    ) -> Dict[str, o3d.geometry.PointCloud]:
        """
        Recenters and constructs a dictionary of pedestrian PCDs keyed by a unique identifier.

        Args:
            scenario_id (str): Scenario ID.
            frame_id (str): Frame ID.
            df_pedestrians_filtered (pd.DataFrame): Filtered DataFrame containing pedestrians.
            pedestrian_pcds_filtered (list): List of filtered pedestrian point clouds.

        Returns:
            Dict[str, o3d.geometry.PointCloud]: A dictionary mapping unique keys to pedestrian PCD objects.
        """
        pedestrian_pcd_dict = {}

        for ped_track_id, pcd in zip(df_pedestrians_filtered['track_id'], pedestrian_pcds_filtered):
            points = np.asarray(pcd.points)
            centroid = points.mean(axis=0)
            points_recentered = points - centroid
            pcd.points = o3d.utility.Vector3dVector(points_recentered)

            pedestrian_key = f"{scenario_id}_{frame_id}_ped_{ped_track_id}"
            pedestrian_pcd_dict[pedestrian_key] = pcd

        return pedestrian_pcd_dict

    def save_pedestrian_pcds(self, pcd_dict: Dict[str, o3d.geometry.PointCloud]) -> None:
        """
        Saves the pedestrian point clouds in the dictionary as .ply files asynchronously.

        Args:
            pcd_dict (Dict[str, o3d.geometry.PointCloud]): Dictionary mapping unique keys to PCD objects.
        """

        def _save_thread():
            for key, pcd in pcd_dict.items():
                filename = f"{key}.ply"
                filepath = os.path.join(self.save_dir, filename)
                try:
                    o3d.io.write_point_cloud(filepath, pcd)
                    self.logger.info(f"Saved pedestrian PCD: {filepath}")
                except Exception as e:
                    self.logger.error(f"Error saving {filename}: {e}")
            self.logger.info("All pedestrian point clouds for the current frame have been saved.")

        # Start the saving process in a new thread.
        thread = threading.Thread(target=_save_thread)
        thread.start()
        self.logger.info("Started async save of pedestrian PCDs.")
