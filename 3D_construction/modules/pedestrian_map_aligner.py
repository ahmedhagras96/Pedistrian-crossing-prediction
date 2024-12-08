import os

import open3d as o3d
import numpy as np

from .base_aligner import BaseAligner
from .utils.logger import Logger
from .utils.pointcloud_utils import PointCloudUtils
from .utils.recon_3d_config import Reconstuction3DConfig


class PedestrianMapAligner(BaseAligner):
    """
    Aligns and processes pedestrian and car data in a point cloud scenario.

    This class manages alignment tasks for pedestrian and car point clouds, including 
    filtering relevant frames, extracting data, and applying transformations. It also provides 
    functionality to crop and save point cloud data.
    """

    def __init__(self, scenario_path: str, loki_csv_path: str, num_frames: int, pedestrian_id: str):
        """
        Initializes the PedestrianMapAligner instance.

        Args:
            scenario_path (str): Path to the scenario data directory.
            loki_csv_path (str): Path to the LOKI CSV file containing metadata.
            num_frames (int): Number of frames to process.
            pedestrian_id (str): ID of the pedestrian to track and align.

        Notes:
            - Initializes necessary attributes and logs the setup.
            - Scenario name is derived from the LOKI scenario ID.
        """
        super().__init__(scenario_path, loki_csv_path, num_frames)

        self.num_frames = num_frames
        self.map_pcd = None
        self.frames = None
        self.pedestrian_id = pedestrian_id
        self.scenario_name = f"scenario_{self.loki.scenario_id}"
        self.pedestrian_data = {}
        self.car_data = {}
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def align(self, *args):
        """
        Aligns and processes the pedestrian and car data within the given frames.

        This method filters relevant frames and extracts the pedestrian and car point cloud data 
        from a scenario. It applies necessary transformations based on odometry data and generates 
        scaled bounding boxes and point clouds for the pedestrian and cars.

        Args:
            *args: Additional arguments for extensibility (currently not used).

        Returns:
            tuple:
                - map_pcd (o3d.geometry.PointCloud): The point cloud of the map from the scenario.
                - ped_ply (o3d.geometry.PointCloud): The point cloud of the target pedestrian.
                - cars_ply (o3d.geometry.PointCloud): The combined point cloud of all cars in the last frame.
                - scaled_box (o3d.geometry.OrientedBoundingBox): The scaled bounding box of the pedestrian.

        Raises:
            KeyError: If expected columns are missing from the `label3d` DataFrame.
            ValueError: If transformations fail or invalid data is encountered during alignment.

        Notes:
            - Assumes odometry data, labels, and point clouds are available and correctly formatted.
            - Objects of interest are hardcoded as 'Car' and 'Pedestrian'. Adjust `objects_needed` 
              for different use cases.
            - The pedestrian's scaled bounding box is scaled by a factor of 15.
         """

        self.frames = self._get_relevant_frames()

        for frame in self.frames:
            pcd, odom, label3d = self.loki.load_alignment_data(frame)
            objects = label3d[label3d['labels'].isin(Reconstuction3DConfig.tracked_objects)]
            target_pedestrian_exists = objects[objects['track_id'].isin([self.pedestrian_id])].shape[0] > 0
            if target_pedestrian_exists:
                transformation_matrix = self.get_transformation_matrix(odom)

                for _, obj_row in objects.iterrows():
                    dimensions = [float(obj_row[col]) for col in obj_row.index[3:10]]
                    center_box, yaw, (l, w, h) = dimensions[:3], dimensions[6], dimensions[3:6]
                    yaw_matrix = PointCloudUtils.get_yaw_matrix(yaw)

                    bounding_box_o3d = o3d.geometry.OrientedBoundingBox(center_box, yaw_matrix, [l, w, h])
                    points_ix = bounding_box_o3d.get_point_indices_within_bounding_box(pcd.points)
                    obj_ply = pcd.select_by_index(points_ix)
                    obj_ply.transform(transformation_matrix)
                    obj_points = np.asarray(obj_ply.points)
                    scaled_box = bounding_box_o3d.scale(25, bounding_box_o3d.get_center())

                    if obj_row['labels'] == 'Pedestrian' and obj_row['track_id'] == self.pedestrian_id:
                        self.pedestrian_data[frame] = obj_points, scaled_box
                    elif obj_row['labels'] == 'Car':
                        self.car_data.setdefault(frame, []).append(obj_points)

        last_frame = self.frames[-1]
        ped_points, bounding_box_scaled = self.pedestrian_data[last_frame]
        car_points = self.car_data[last_frame]
        ped_ply = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ped_points))
        cars_ply = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(car_points, axis=0)))
        self.map_pcd = o3d.io.read_point_cloud(os.path.join(self.scenario_path, 'map.ply'))
        return self.map_pcd, ped_ply, cars_ply, bounding_box_scaled

    def save(self, save_path: str, remove: bool = False):
        """
        Saves the cropped pedestrian point clouds.

        Args:
            save_path (str): Directory path where the cropped point clouds will be saved.
            remove (bool): Whether to remove the cropped region from the original map.

        Notes:
            - Creates directories if they do not exist.
            - Saves point clouds for each processed frame in the specified directory.
            - Each frame's point cloud is saved in a separate file named `frame_<frame_number>.ply`.
        """
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)
        path_ped = os.path.join(save_path, f"ped_{self.pedestrian_id}")
        os.makedirs(path_ped, exist_ok=True)

        for frame in self.frames:
            try:
                # Crop the point cloud for the pedestrian in the current frame
                cropped_pcd = self._crop(frame, remove)

                # Ensure the cropped point cloud is a valid Open3D PointCloud object
                if isinstance(cropped_pcd, np.ndarray):
                    cropped_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cropped_pcd))
                elif not isinstance(cropped_pcd, o3d.geometry.PointCloud):
                    raise TypeError(f"Invalid point cloud type: {type(cropped_pcd)}")

                # Save the cropped point cloud to a file
                file_path = os.path.join(path_ped, f"frame_{frame}.ply")
                success = o3d.io.write_point_cloud(file_path, cropped_pcd, write_ascii=True)
                if not success:
                    raise RuntimeError(f"Failed to write point cloud to {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to process frame {frame}: {e}")

        self.logger.info(f"Saved cropped pedestrian point clouds to: {path_ped}")

    def _get_relevant_frames(self):
        """
        Retrieves relevant frame numbers from the LOKI CSV.

        Filters rows that match the current scenario and pedestrian ID. Limits the 
        number of extracted frames to the specified `num_frames`.

        Returns:
            list: A list of relevant frame numbers for processing.

        Notes:
            - Logs warnings if no data is found for the scenario or pedestrian ID.
            - Handles invalid frame name formats gracefully.
        """
        self.logger.info("Loading LOKI CSV data.")
        loki_data = self.loki.load_loki_csv()
        filtered_scenario_data = loki_data[loki_data['video_name'] == self.scenario_name]

        if filtered_scenario_data.empty:
            self.logger.warning(f"No data found for scenario: {self.scenario_name}")
            return []

        self.logger.info(f"Filtering data for Pedestrian ID: {self.pedestrian_id} in scenario: {self.scenario_name}.")
        filtered_pedestrian_data = filtered_scenario_data[filtered_scenario_data['Ped_ID'] == self.pedestrian_id]

        if filtered_pedestrian_data.empty:
            self.logger.warning(
                f"No data found for Pedestrian ID: {self.pedestrian_id} in scenario: {self.scenario_name}.")
            return []

        frame_numbers = []
        frame_count = 0

        self.logger.info(f"Extracting up to {self.num_frames} frame number(s).")
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
        return frame_numbers

    def _crop(self, frame: int, remove: bool):
        """
        Crops the map point cloud to the pedestrian's bounding box.

        Args:
            frame (int): Frame number to crop.
            remove (bool): Whether to remove the cropped region from the original map.

        Returns:
            o3d.geometry.PointCloud: The cropped point cloud.

        Notes:
            - Uses the pedestrian's bounding box data from the specified frame.
        """
        self.logger.info(f"Cropping the map to the bounding box of the pedestrian in frame: {frame}.")
        ped_points, bounding_box_o3d = self.pedestrian_data[frame]
        car_points = self.car_data[frame]
        concat_points = np.concatenate(
            [ped_points, np.concatenate(car_points, axis=0), np.asarray(self.map_pcd.points)], axis=0)
        concat_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(concat_points))
        cropped_pcd = concat_pcd.crop(bounding_box_o3d, invert=remove)
        return cropped_pcd
