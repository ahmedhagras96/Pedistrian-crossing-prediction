import os
from typing import Dict, Tuple


class PositionLoader:
    """
    Utility class to load pedestrian and vehicle positions from files.
    """

    @staticmethod
    def load_label3d_positions(scenario_folder: str) -> Dict[int, Dict[str, Tuple[float, float, float]]]:
        """
        Load 3D positions of pedestrians for each frame in a scenario.

        Args:
            scenario_folder (str): Path to the scenario folder.

        Returns:
            dict: Frame-wise pedestrian positions.
        """
        label3d_files = sorted([
            f for f in os.listdir(scenario_folder)
            if f.startswith("label3d_") and f.endswith(".txt")
        ])
        pedestrian_positions_frames: Dict[int, Dict[str, Tuple[float, float, float]]] = {}

        for file in label3d_files:
            frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))
            pedestrian_positions_frames[frame_id] = {}
            with open(os.path.join(scenario_folder, file), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(',')
                    if parts[0] == "Pedestrian":
                        ped_id = parts[1]
                        pos_x, pos_y, pos_z = map(float, parts[3:6])
                        pedestrian_positions_frames[frame_id][ped_id] = (pos_x, pos_y, pos_z)

        return pedestrian_positions_frames

    @staticmethod
    def load_vehicle_positions(scenario_folder: str) -> Dict[int, Tuple[float, float, float]]:
        """
        Load vehicle positions for each frame in a scenario.

        Args:
            scenario_folder (str): Path to the scenario folder.

        Returns:
            dict: Frame-wise vehicle positions.
        """
        odom_files = sorted([
            f for f in os.listdir(scenario_folder)
            if f.startswith("odom_") and f.endswith(".txt")
        ])
        vehicle_positions: Dict[int, Tuple[float, float, float]] = {}

        for file in odom_files:
            frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))
            with open(os.path.join(scenario_folder, file), 'r') as f:
                line = f.readline().strip()
                x, y, z, *_ = map(float, line.split(','))
                vehicle_positions[frame_id] = (x, y, z)

        return vehicle_positions
