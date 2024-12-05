import os

import open3d as o3d
import numpy as np

from .base_aligner import BaseAligner
from .utils.logger import Logger
from .utils.pointcloud_utils import PointCloudUtils
from .utils.visualization import PointCloudVisualizer

class PedestrianMapAligner(BaseAligner):
    def __init__(self, scenario_path: str, loki_csv_path: str, num_frames: int, pedestrian_id: str):
        super().__init__(scenario_path, loki_csv_path, num_frames)
        
        self.num_frames = num_frames
        self.pedestrian_id = pedestrian_id
        self.scenario_name = f"scenario_{self.loki.scenario_id}"
        self.obj = {}
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def align(self, *args):
        frames = self._get_relevant_frames()
        objects_needed = ['Car', 'Pedestrian']  # TODO: pass this value
        
        for frame in frames:
            pcd, odom, label3d = self.loki.load_aligment_data(frame)
            objects_needed = ['Car', 'Pedestrian']  # TODO: pass this value
            objects = label3d[label3d['labels'].isin(objects_needed)]
            target_pedestrian_exists = (objects[objects['track_id'].isin([self.pedestrian_id])]).any()
            # if not target_pedestrian_exists:
            #     # log/throw error
            #     return
            
            # log confirmation

            for _, object_data in objects.iterrows():  # Use `_` for the index since it's not needed
                position = (object_data['pos_x'], object_data['pos_y'], object_data['pos_z'])
                dimensions = (object_data['dim_x'], object_data['dim_y'], object_data['dim_z'])
                yaw = PointCloudUtils.get_yaw_matrix(float(object_data['yaw']))
            
                # print(f"Position: {position}, Dimensions: {dimensions}, Yaw: {yaw}")

        
                bounding_box_o3d = o3d.geometry.OrientedBoundingBox(position, yaw, dimensions)
                # self.logger.debug(f"Bounding box created for frame {frame}: center={position}, dimensions={dimensions}, yaw={yaw}")
                
                pcd = self.loki.load_ply(frame)
                points_ix = bounding_box_o3d.get_point_indices_within_bounding_box(pcd.points)
                obj_ply = pcd.select_by_index(points_ix)
                obj_points = np.asarray(obj_ply.points)
                
                odom = self.loki.load_odometry(frame)
                
                # TODO: CHECK IF THIS IS CORRECT COMPARED TO LEGACY CODE
                transformation_matrix = self.get_transformation_matrix(odom)
                obj_ply.transform(transformation_matrix)
                self.obj[frame] = obj_ply

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.array([0.2, 0.2, 0.2])  # Set background color
        vis.get_render_option().point_size = 2.0  # Set point size
        last_frame = frames[-1]
        ped_data = self.obj[last_frame]

        # Adding a single pedestrain based on its ID
        # ped_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ped_data))
        # ped_pcd.paint_uniform_color([0.5, 0.5, 1])
        vis.add_geometry(ped_data)

        # Adding the bounding box of the pedestrian
        map_pcd = o3d.io.read_point_cloud(os.path.join(self.scenario_path, 'map.ply'))
        # vis.add_geometry(map_pcd)
        
        visualizer = PointCloudVisualizer()
        visualizer.visualize(map_pcd)
        visualizer.visualize(ped_data)        


    def _get_relevant_frames(self):
            # Load the LOKI CSV and filter rows matching the current scenario
            self.logger.info("Loading LOKI CSV data.")
            loki_data = self.loki.load_loki_csv()
            filtered_scenario_data = loki_data[loki_data['video_name'] == self.scenario_name]
    
            if filtered_scenario_data.empty:
                self.logger.warning(f"No data found for scenario: {self.scenario_name}")
                return []
    
            self.logger.info(f"Filtering data for Pedestrian ID: {self.pedestrian_id} in scenario {self.scenario_name}.")
            filtered_pedestrian_data = filtered_scenario_data[filtered_scenario_data['Ped_ID'] == self.pedestrian_id]
    
            if filtered_pedestrian_data.empty:
                self.logger.warning(f"No data found for Pedestrian ID: {self.pedestrian_id} in scenario {self.scenario_name}.")
                return []
    
            frame_numbers = []
            frame_count = 0
    
            self.logger.info(f"Extracting up to {self.num_frames} frame numbers.")
            for _, row in filtered_pedestrian_data.iterrows():
                frame_name = row['frame_name']
                try:
                    frame_number = int(frame_name.split('_')[-1])  # Ensure we extract numeric frame numbers
                except ValueError:
                    self.logger.error(f"Invalid frame name format: {frame_name}. Skipping.")
                    continue
    
                frame_numbers.append(frame_number)
                frame_count += 1
    
                if frame_count >= self.num_frames:
                    self.logger.info(f"Reached frame limit: {self.num_frames}. Stopping extraction.")
                    break
    
            self.logger.info(f"Extracted {len(frame_numbers)} frame numbers: {frame_numbers}.")
            return sorted(frame_numbers)
    
            