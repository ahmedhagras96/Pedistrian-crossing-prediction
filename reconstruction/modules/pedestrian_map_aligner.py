import os
import time
import open3d as o3d
import numpy as np
from typing import Dict, List

from .base_aligner import BaseAligner
from .utils.logger import Logger
from .utils.pointcloud_utils import PointCloudUtils
from .utils.recon_3d_config import Reconstuction3DConfig
from .utils.threading import ThreadedSaveManager


class PedestrianMapAligner(BaseAligner):
    """
    Aligns and processes pedestrian and car data in a point cloud scenario.
    """

    def __init__(self, scenario_path: str, loki_csv_path: str, data_path: str):
        """
        Initializes the PedestrianMapAligner with the given paths.

        Args:
            scenario_path (str): Path to the scenario data.
            loki_csv_path (str): Path to the LOKI CSV data.
            data_path (str): Path to the data folder.
        """
        super().__init__(scenario_path, loki_csv_path)
        self.loki_folder_path = data_path
        self.save_manager = ThreadedSaveManager()
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def align(
        self, 
        save: bool = False, 
        use_downsampling: bool = False,
        save_path: str = None, 
        scaling_factor: int = 25
    ) -> None:
        """
        Aligns and processes the pedestrian and car data within the given frames.

        Args:
            save (bool): Whether to save the processed data.
            use_downsampling (bool): Whether to use downsampling on the point cloud.
            save_path (str): Path to save the processed point cloud.
            scaling_factor (int): Factor to scale the bounding boxes.

        Returns:
            None
        """
        self.use_downsampling = use_downsampling
        scenario_frames = self._get_relevant_frames()
        total_save_time = 0
        start_time = time.time()

        for scenario_name, frame_to_peds in scenario_frames.items():
            self.scenario_name = scenario_name
            scenario_path = os.path.join(self.loki_folder_path, scenario_name)
            self.map_pcd = o3d.io.read_point_cloud(os.path.join(scenario_path, 'map.ply'))

            if self.use_downsampling:
                self.map_pcd = self.map_pcd.voxel_down_sample(voxel_size=Reconstuction3DConfig.voxel_size)

            for frame, ped_ids in frame_to_peds.items():
                self.logger.info(f"Processing frame {frame} for pedestrians {ped_ids}")
                pcd, odom, label3d = self.loki.load_alignment_data(frame, scenario_path)

                if label3d.empty:
                    self.logger.warning(f"No labels found for frame {frame} in scenario {self.scenario_name}")
                    continue

                transformation_matrix = self.get_transformation_matrix(odom)
                objects = label3d[label3d['labels'].isin(Reconstuction3DConfig.tracked_objects)]

                # Process pedestrian data
                for ped_id in ped_ids:
                    ped_objects = objects[objects['track_id'] == ped_id]
                    if ped_objects.empty:
                        self.logger.warning(f"No pedestrian with id {ped_id} found in frame {frame}")
                        continue

                    for _, obj_row in ped_objects.iterrows():
                        dimensions = [float(obj_row[col]) for col in obj_row.index[3:10]]
                        center_box, yaw, (l, w, h) = dimensions[:3], dimensions[6], dimensions[3:6]
                        yaw_matrix = PointCloudUtils.get_yaw_matrix(yaw)
                        bounding_box_o3d = o3d.geometry.OrientedBoundingBox(center_box, yaw_matrix, [l, w, h])
                        scaled_box = bounding_box_o3d.scale(scaling_factor, bounding_box_o3d.get_center())

                        points_ix = bounding_box_o3d.get_point_indices_within_bounding_box(pcd.points)
                        obj_ply = pcd.select_by_index(points_ix)
                        obj_ply.transform(transformation_matrix)

                        if obj_row['labels'] == 'Pedestrian':
                            # Collect car data for the same frame
                            car_objects = objects[objects['labels'] == 'Car']
                            car_points = []
                            for _, car_row in car_objects.iterrows():
                                car_dimensions = [float(car_row[col]) for col in car_row.index[3:10]]
                                car_center_box, car_yaw, (car_l, car_w, car_h) = car_dimensions[:3], car_dimensions[6], car_dimensions[3:6]
                                car_yaw_matrix = PointCloudUtils.get_yaw_matrix(car_yaw)
                                car_bounding_box = o3d.geometry.OrientedBoundingBox(car_center_box, car_yaw_matrix, [car_l, car_w, car_h])
                                car_points_ix = car_bounding_box.get_point_indices_within_bounding_box(pcd.points)
                                car_ply = pcd.select_by_index(car_points_ix)
                                car_ply.transform(transformation_matrix)
                                car_points.append(np.asarray(car_ply.points))

                            # Crop and save the combined point cloud
                            cropped_pcd = self._crop(frame, obj_ply, car_points, scaled_box)
                            if save and cropped_pcd:
                                self.save(save_path, scenario_name, frame, ped_id, cropped_pcd)

        end_time = time.time()
        total_save_time = end_time - start_time
        self.logger.info(f"Total save time: {total_save_time:.2f} seconds")

    def save(self, save_path: str, scenario_name: str, frame: int, pedestrian_id: str, cropped_pcd: o3d.geometry.PointCloud) -> None:
        """
        Saves the cropped point cloud to the specified path.

        Args:
            save_path (str): Path to save the point cloud.
            scenario_name (str): Name of the scenario.
            frame (int): Frame number.
            pedestrian_id (str): ID of the pedestrian.
            cropped_pcd (o3d.geometry.PointCloud): The cropped point cloud to save.

        Returns:
            None
        """
        os.makedirs(save_path, exist_ok=True)
        scenario_id = scenario_name.split('_')[-1]
        file_name = f"{scenario_id}_{frame:04}_ped_{pedestrian_id}.ply"
        file_path = os.path.join(save_path, file_name)

        try:
            self.save_manager.add_save_task(file_path, cropped_pcd)
        except Exception as e:
            self.logger.error(f"Failed to save frame {frame} for pedestrian {pedestrian_id}: {e}")

    def _get_relevant_frames(self) -> Dict[str, Dict[int, List[str]]]:
        """
        Retrieves relevant frame numbers from the LOKI CSV.

        Returns:
            Dict[str, Dict[int, List[str]]]: A dictionary mapping scenario IDs to frame numbers and pedestrian IDs.
        """
        self.logger.info("Loading LOKI CSV data.")
        loki_data = self.loki.load_loki_csv()
        if loki_data.empty:
            self.logger.warning("No data found in loki_data")
            return {}

        scenario_frames = {}
        for _, row in loki_data.iterrows():
            frame_number = row['frame_id']
            scenario_id = f"scenario_{row['scenario_id']:03}"
            ped_id = row['track_id']

            scenario_frames.setdefault(scenario_id, {}).setdefault(frame_number, []).append(ped_id)

        total_peds = sum(len(peds) for peds in scenario_frames.values())
        total_frames = sum(len(frames) for peds in scenario_frames.values() for frames in peds.values())
        self.logger.info(f"Extracted {total_peds} pedestrians in {total_frames} frames across all scenarios.")

        return scenario_frames

    def _crop(
        self, 
        frame: int, 
        obj_ply: o3d.geometry.PointCloud, 
        car_points: List[np.ndarray], 
        bounding_box_o3d: o3d.geometry.OrientedBoundingBox
    ) -> o3d.geometry.PointCloud:
        """
        Crops the map point cloud to the pedestrian's bounding box, including car data.

        Args:
            frame (int): Frame number.
            obj_ply (o3d.geometry.PointCloud): The point cloud of the object to crop.
            car_points (List[np.ndarray]): List of car point clouds.
            bounding_box_o3d (o3d.geometry.OrientedBoundingBox): The bounding box for cropping.

        Returns:
            o3d.geometry.PointCloud: The cropped point cloud.
        """
        if bounding_box_o3d.is_empty() or self.map_pcd.is_empty():
            self.logger.warning(f"Skipping cropping for frame {frame} due to empty bounding box or map point cloud.")
            return None

        # Combine pedestrian, car, and map points
        concat_points = np.concatenate(
            [np.asarray(obj_ply.points), np.concatenate(car_points, axis=0), np.asarray(self.map_pcd.points)], 
            axis=0
        )
        concat_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(concat_points))

        cropped_pcd = concat_pcd.crop(bounding_box_o3d, invert=False)

        self.logger.info(f"Finished cropping, {len(cropped_pcd.points)} points for frame: {frame}.")
        return cropped_pcd
