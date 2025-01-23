import os
import time
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d

from modules.config.logger import Logger
from modules.reconstruction.aligners.base_aligner import BaseAligner
from modules.reconstruction.utilities.pointcloud_utils import PointCloudUtils
from modules.reconstruction.utilities.recon_3d_config import Reconstuction3DConfig
from modules.reconstruction.utilities.threaded_saving import ThreadedPointCloudSaver


class PedestrianMapAligner(BaseAligner):
    """
    Aligns and processes pedestrian and car data in a point cloud scenario.
    """

    def __init__(self, loki_csv_path: str, data_path: str, scenario_path: Optional[str] = None):
        """
        Initialize the PedestrianMapAligner with the given paths.

        Args:
            loki_csv_path (str): Path to the LOKI CSV data.
            data_path (str): Path to the data folder.
            scenario_path (str, optional): Path to the scenario data.

        Returns:
            None
        """
        super().__init__(scenario_path, loki_csv_path)
        self.loki_folder_path = data_path
        self.save_manager = ThreadedPointCloudSaver()
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

        # Control whether to downsample the point clouds
        self.use_downsampling: bool = False

    def align(
            self,
            save: bool = False,
            use_downsampling: bool = False,
            save_path: Optional[str] = None,
            scaling_factor: int = 25
    ) -> None:
        """
        Align and process pedestrian and car data for each relevant scenario frame.

        Args:
            save (bool): Whether to save the processed data.
            use_downsampling (bool): Whether to downsample the point cloud.
            save_path (str, optional): Directory to save the processed point clouds.
            scaling_factor (int): Factor by which to scale the bounding boxes.

        Returns:
            None
        """
        self.use_downsampling = use_downsampling
        scenario_frames = self._get_relevant_frames()

        start_time = time.time()

        for scenario_name, frame_to_peds in scenario_frames.items():
            # Load map point cloud for the scenario
            scenario_path = os.path.join(self.loki_folder_path, scenario_name)
            map_pcd = self._load_scenario_map(scenario_path)

            for frame, ped_ids in frame_to_peds.items():
                self._process_frame(
                    scenario_name=scenario_name,
                    scenario_path=scenario_path,
                    frame=frame,
                    ped_ids=ped_ids,
                    map_pcd=map_pcd,
                    scaling_factor=scaling_factor,
                    save=save,
                    save_path=save_path
                )

        total_time = time.time() - start_time
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")

    def _load_scenario_map(self, scenario_path: str) -> o3d.geometry.PointCloud:
        """
        Load the map point cloud for a scenario and optionally downsample it.

        Args:
            scenario_path (str): The directory containing the scenario data.

        Returns:
            o3d.geometry.PointCloud: The loaded and (optionally) downsampled point cloud.
        """
        map_path = os.path.join(scenario_path, 'map.ply')
        map_pcd = o3d.io.read_point_cloud(map_path)

        if self.use_downsampling:
            map_pcd = map_pcd.voxel_down_sample(voxel_size=Reconstuction3DConfig.voxel_size)
            self.logger.info(f"Downsampled map point cloud at {scenario_path}")

        return map_pcd

    def _process_frame(
            self,
            scenario_name: str,
            scenario_path: str,
            frame: int,
            ped_ids: List[str],
            map_pcd: o3d.geometry.PointCloud,
            scaling_factor: int,
            save: bool,
            save_path: Optional[str]
    ) -> None:
        """
        Process a single frame within a scenario by transforming point clouds
        and cropping them around pedestrians (and nearby cars).

        Args:
            scenario_name (str): Name of the scenario.
            scenario_path (str): Path to the scenario data on disk.
            frame (int): Frame number.
            ped_ids (List[str]): List of pedestrian IDs for this frame.
            map_pcd (o3d.geometry.PointCloud): Pre-loaded map point cloud.
            scaling_factor (int): Factor to scale bounding boxes.
            save (bool): Whether to save the cropped point clouds.
            save_path (str, optional): Directory in which to save the processed point clouds.

        Returns:
            None
        """
        self.logger.info(f"Processing frame {frame} (scenario: {scenario_name}) for pedestrians: {ped_ids}")
        pcd, odom, label3d = self.loki.load_alignment_data(frame, scenario_path)

        if label3d.empty:
            self.logger.warning(f"No labels found for frame {frame} in scenario {scenario_name}")
            return

        transformation_matrix = self.get_transformation_matrix(odom)
        translation = transformation_matrix[:3, 3]
        rotation = transformation_matrix[:3, :3]

        objects = label3d[label3d['labels'].isin(Reconstuction3DConfig.tracked_objects)]
        for ped_id in ped_ids:
            self._process_pedestrian(
                ped_id=ped_id,
                objects=objects,
                pcd=pcd,
                map_pcd=map_pcd,
                transformation_matrix=transformation_matrix,
                translation=translation,
                rotation=rotation,
                frame=frame,
                scenario_name=scenario_name,
                scaling_factor=scaling_factor,
                save=save,
                save_path=save_path
            )

    def _process_pedestrian(
            self,
            ped_id: str,
            objects,
            pcd: o3d.geometry.PointCloud,
            map_pcd: o3d.geometry.PointCloud,
            transformation_matrix: np.ndarray,
            translation: np.ndarray,
            rotation: np.ndarray,
            frame: int,
            scenario_name: str,
            scaling_factor: int,
            save: bool,
            save_path: Optional[str]
    ) -> None:
        """
        Process a single pedestrian by extracting the pedestrian point cloud,
        collecting nearby car data, and cropping the result.

        Args:
            ped_id (str): The ID of the pedestrian to process.
            objects (pandas.DataFrame): All detected objects (pandas DataFrame).
            pcd (o3d.geometry.PointCloud): The untransformed point cloud for this frame.
            map_pcd (o3d.geometry.PointCloud): The map point cloud for this scenario.
            transformation_matrix (np.ndarray): 4x4 transformation matrix from odometry.
            translation (np.ndarray): Translation component of the transformation.
            rotation (np.ndarray): Rotation component of the transformation.
            frame (int): Frame number.
            scenario_name (str): Name of the scenario.
            scaling_factor (int): Scaling factor for the bounding box.
            save (bool): Whether to save the resulting point cloud.
            save_path (str, optional): Directory to save the cropped point clouds.

        Returns:
            None
        """
        ped_objects = objects[objects['track_id'] == ped_id]
        if ped_objects.empty:
            self.logger.warning(f"No pedestrian with id {ped_id} found in frame {frame}")
            return

        for _, obj_row in ped_objects.iterrows():
            # Extract bounding box data
            dimensions = [float(obj_row[col]) for col in obj_row.index[3:10]]
            center_box, yaw, (l, w, h) = dimensions[:3], dimensions[6], dimensions[3:6]

            yaw_matrix = PointCloudUtils.get_yaw_matrix(yaw)
            bounding_box_o3d = o3d.geometry.OrientedBoundingBox(center_box, yaw_matrix, [l, w, h])

            # Exctract Pedestrian points from the point cloud 
            points_ix = bounding_box_o3d.get_point_indices_within_bounding_box(pcd.points)
            self.logger.info(f"Extracted {len(points_ix)} points for pedestrian {ped_id} in frame {frame}")
            obj_ply = pcd.select_by_index(points_ix)

            # Apply transformation on the pedestrian point cloud
            obj_ply.transform(transformation_matrix)

            # Scale, translate, and rotate the bounding box
            scaled_box = bounding_box_o3d.scale(scaling_factor, bounding_box_o3d.get_center())
            scaled_box.translate(translation)
            scaled_box.rotate(rotation)

            # If this label is a pedestrian, gather car data
            if obj_row['labels'] == 'Pedestrian':
                car_points = self._collect_car_points(objects, pcd)
                cropped_pcd = self._crop(
                    frame=frame,
                    obj_ply=obj_ply,
                    map_pcd=map_pcd,
                    car_points=car_points,
                    bounding_box_o3d=scaled_box
                )

                if save and cropped_pcd is not None and save_path:
                    self.save(save_path, scenario_name, frame, ped_id, cropped_pcd)

    def _collect_car_points(self, objects, pcd: o3d.geometry.PointCloud) -> List[np.ndarray]:
        """
        Collect and return the points corresponding to all 'Car' objects in this frame.

        Args:
            objects (pandas.DataFrame): DataFrame containing object labels for this frame.
            pcd (o3d.geometry.PointCloud): The (already transformed) point cloud for this frame.

        Returns:
            List[np.ndarray]: A list of NumPy arrays representing car points.
        """
        car_objects = objects[objects['labels'] == 'Car']
        car_points = []

        for _, car_row in car_objects.iterrows():
            car_dimensions = [float(car_row[col]) for col in car_row.index[3:10]]
            center_box, car_yaw, (car_l, car_w, car_h) = car_dimensions[:3], car_dimensions[6], car_dimensions[3:6]
            car_yaw_matrix = PointCloudUtils.get_yaw_matrix(car_yaw)

            car_bounding_box = o3d.geometry.OrientedBoundingBox(
                center_box, car_yaw_matrix, [car_l, car_w, car_h]
            )
            car_points_ix = car_bounding_box.get_point_indices_within_bounding_box(pcd.points)
            car_ply = pcd.select_by_index(car_points_ix)
            car_points.append(np.asarray(car_ply.points))

        return car_points

    def _crop(
            self,
            frame: int,
            obj_ply: o3d.geometry.PointCloud,
            map_pcd: o3d.geometry.PointCloud,
            car_points: List[np.ndarray],
            bounding_box_o3d: o3d.geometry.OrientedBoundingBox
    ) -> Optional[o3d.geometry.PointCloud]:
        """
        Crop the combined point cloud (pedestrian, cars, map) to the bounding box.

        Args:
            frame (int): Frame number.
            obj_ply (o3d.geometry.PointCloud): The point cloud of the target object (e.g., pedestrian).
            map_pcd (o3d.geometry.PointCloud): The map point cloud for this scenario.
            car_points (List[np.ndarray]): List of NumPy arrays representing the points of each car.
            bounding_box_o3d (o3d.geometry.OrientedBoundingBox): The bounding box 
                (already translated/rotated).

        Returns:
            Optional[o3d.geometry.PointCloud]: The cropped point cloud, 
            or None if the bounding box or map is empty.
        """
        if bounding_box_o3d.is_empty() or map_pcd.is_empty():
            self.logger.warning(
                f"Skipping cropping for frame {frame} due to empty bounding box or map point cloud."
            )
            return None

        combined_points = [
            np.asarray(obj_ply.points),
            *car_points,
            np.asarray(map_pcd.points)
        ]
        concat_points = np.concatenate(combined_points, axis=0)
        concat_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(concat_points))

        cropped_pcd = concat_pcd.crop(bounding_box_o3d, invert=False)
        self.logger.info(f"Finished cropping {len(cropped_pcd.points)} points for frame {frame}.")
        return cropped_pcd

    def save(
            self,
            save_path: str,
            scenario_name: str,
            frame: int,
            pedestrian_id: str,
            cropped_pcd: o3d.geometry.PointCloud
    ) -> None:
        """
        Save the cropped point cloud to the specified path using a threaded manager.

        Args:
            save_path (str): Directory to save the point cloud.
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
        Retrieve a mapping of scenario -> frame -> list of pedestrian IDs from the LOKI CSV.

        Returns:
            Dict[str, Dict[int, List[str]]]: A dictionary mapping scenario IDs to another 
            dictionary of frame numbers mapped to lists of pedestrian IDs.
        """
        self.logger.info("Loading LOKI CSV data.")
        loki_data = self.loki.load_loki_csv()
        if loki_data.empty:
            self.logger.warning("No data found in LOKI CSV.")
            return {}

        scenario_frames: Dict[str, Dict[int, List[str]]] = {}
        for _, row in loki_data.iterrows():
            frame_number = row['frame_id']
            scenario_id = f"scenario_{row['scenario_id']:03}"
            ped_id = row['track_id']

            scenario_frames.setdefault(scenario_id, {}).setdefault(frame_number, []).append(ped_id)

        total_peds = sum(len(peds) for peds_dict in scenario_frames.values() for peds in peds_dict.values())
        total_frames = sum(len(peds_dict) for peds_dict in scenario_frames.values())
        self.logger.info(
            f"Extracted {total_peds} pedestrians in {total_frames} frames across all scenarios."
        )

        return scenario_frames
