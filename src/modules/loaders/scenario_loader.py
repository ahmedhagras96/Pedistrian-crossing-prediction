import os

from modules.utilities.logger import LoggerUtils


class ScenarioLoader:
    def __init__(self):
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

    def load_label3d_positions(self, scenario_folder):
        """
        Load pedestrian positions from label3d text files in a scenario.
    
        Args:
            scenario_folder (str): Directory containing label3d files.
    
        Returns:
            dict: Mapping of frame IDs to pedestrian positions.
        """
        self.logger.info(f"Loading label3d positions from {scenario_folder}")
        label3d_files = sorted(
            [f for f in os.listdir(scenario_folder) if f.startswith("label3d_") and f.endswith(".txt")])
        pedestrian_positions_frames = {}

        for file in label3d_files:
            frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))
            pedestrian_positions_frames[frame_id] = {}
            path = os.path.join(scenario_folder, file)
            with open(path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(',')
                    if parts[0] == "Pedestrian":
                        ped_id = parts[1]
                        pos_x, pos_y, pos_z = map(float, parts[3:6])
                        pedestrian_positions_frames[frame_id][ped_id] = (pos_x, pos_y, pos_z)
        return pedestrian_positions_frames

    def load_vehicle_positions(self, scenario_folder):
        """
        Load vehicle positions from odom text files in a scenario.
    
        Args:
            scenario_folder (str): Directory containing odom files.
    
        Returns:
            dict: Mapping of frame IDs to vehicle positions.
        """
        self.logger.info(f"Loading vehicle positions from {scenario_folder}")
        odom_files = sorted([f for f in os.listdir(scenario_folder) if f.startswith("odom_") and f.endswith(".txt")])
        vehicle_positions = {}

        for file in odom_files:
            frame_id = int(''.join(filter(str.isdigit, file.split('_')[1].split('.')[0])))
            path = os.path.join(scenario_folder, file)
            with open(path, 'r') as f:
                line = f.readline().strip()
                values = list(map(float, line.split(',')))
                vehicle_positions[frame_id] = tuple(values[:3])
        return vehicle_positions
