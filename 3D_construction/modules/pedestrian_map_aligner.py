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

    def __init__(self, scenario_path: str, loki_csv_path: str, loki_folder_path: str):
        """
        Initializes the PedestrianMapAligner instance.

        Args:
            scenario_path (str): Path to the scenario data directory.
            loki_csv_path (str): Path to the LOKI CSV file containing metadata.
            loki_folder_path (str): path to all of the scenario folders.

        Notes:
            - Initializes necessary attributes and logs the setup.
            - Scenario name is derived from the LOKI scenario ID.
        """
        super().__init__(scenario_path, loki_csv_path)

        self.num_frames = None
        self.loki_folder_path = loki_folder_path
        self.map_pcd = None
        self.frames = None
        self.pedestrian_id = None
        self.scenario_name = None 
        self.pedestrian_data = {}
        self.car_data = {}
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def align(
        self, 
        save: bool = False, use_downsampling: bool = False,
        save_path: str = None, scale: int = 25
        ):
        """
        Aligns and processes the pedestrian and car data within the given frames.

        This method filters relevant frames and extracts the pedestrian and car point cloud data 
        from a scenario. It applies necessary transformations based on odometry data and generates 
        scaled bounding boxes and point clouds for the pedestrian and cars.

        Args:
            save (bool): Indicates whether to save the cropped point clouds.
            save_path (str): Directory path where the cropped point clouds will be saved.
            scale (int): Factor to scale the bounding boxes for cropping.

        Raises:
            KeyError: If expected columns are missing from the `label3d` DataFrame.
            ValueError: If transformations fail or invalid data is encountered during alignment.

        Notes:
            - Assumes odometry data, labels, and point clouds are available and correctly formatted.
            - Objects of interest are hardcoded as 'Car' and 'Pedestrian'. Adjust `objects_needed` 
              for different use cases.
            - The pedestrian's scaled bounding box is scaled by a factor of 15.
         """
        self.use_downsampling = use_downsampling
        scenario_to_ped_to_frames = self._get_relevant_frames()
        
        for scenario_name, ID_to_FR in scenario_to_ped_to_frames.items():
            self.scenario_name = scenario_name
            scenario_path = os.path.join(self.loki_folder_path, scenario_name)
            self.map_pcd = o3d.io.read_point_cloud(os.path.join(scenario_path, 'map.ply'))
            for pedestrian_id, frame_sequence in ID_to_FR.items():  
                self.logger.info(f"Processing pedestrian {pedestrian_id}")

                self.pedestrian_id = pedestrian_id
                self.frames = frame_sequence

                for frame in self.frames:
                    pcd, odom, label3d = self.loki.load_alignment_data(frame, scenario_path)

                    # Filter for relevant objects (car and pedestrian)
                    objects = label3d[label3d['labels'].isin(Reconstuction3DConfig.tracked_objects)]

                    target_pedestrian_exists = objects[objects['track_id'].isin([self.pedestrian_id])].shape[0] > 0

                    if target_pedestrian_exists:
                        transformation_matrix = self.get_transformation_matrix(odom)

                        # Iterate over each object in the frame
                        for _, obj_row in objects.iterrows():
                            dimensions = [float(obj_row[col]) for col in obj_row.index[3:10]]
                            center_box, yaw, (l, w, h) = dimensions[:3], dimensions[6], dimensions[3:6]
                            yaw_matrix = PointCloudUtils.get_yaw_matrix(yaw)

                            bounding_box_o3d = o3d.geometry.OrientedBoundingBox(center_box, yaw_matrix, [l, w, h])

                            points_ix = bounding_box_o3d.get_point_indices_within_bounding_box(pcd.points)

                            obj_ply = pcd.select_by_index(points_ix)
                            obj_ply.transform(transformation_matrix)

                            obj_points = np.asarray(obj_ply.points)
                            scaled_box = bounding_box_o3d.scale(scale, bounding_box_o3d.get_center())

                            if obj_row['labels'] == 'Pedestrian' and obj_row['track_id'] == self.pedestrian_id:
                                self.pedestrian_data[frame] = obj_points, scaled_box
                            elif obj_row['labels'] == 'Car':
                                self.car_data.setdefault(frame, []).append(obj_points)          

                # Save the cropped point clouds if requested
                if save:
                    self.save(save_path=save_path)


    def save(self, save_path: str):
        """
        Saves the cropped pedestrian point clouds.

        Args:
            save_path (str): Directory path where the cropped point clouds will be saved.

        Notes:
            - Creates directories if they do not exist.
            - Saves point clouds for each processed frame in the specified directory.
            - Each frame's point cloud is saved in a separate file named `<scenario_id>_<frame_id>_ped_<Ped_ID>.ply`.
        """
        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)

        scenario_id = self.scenario_name.split('_')[-1]
        for frame in self.frames:
            try:
                
                # Crop the point cloud for the pedestrian in the current frame
                cropped_pcd = self._crop(frame)

                # Ensure the cropped point cloud is a valid Open3D PointCloud object
                if isinstance(cropped_pcd, np.ndarray):
                    cropped_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cropped_pcd))
                elif not isinstance(cropped_pcd, o3d.geometry.PointCloud):
                    raise TypeError(f"Invalid point cloud type: {type(cropped_pcd)}")

                # Save the cropped point cloud to a file
                file_path = os.path.join(save_path, f"{scenario_id}_{frame}_ped_{self.pedestrian_id}.ply")
                success = o3d.io.write_point_cloud(file_path, cropped_pcd, write_ascii=True)
                if not success:
                    raise RuntimeError(f"Failed to write point cloud to {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to process frame {frame}: {e}")

 
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

                if loki_data.empty:
                    self.logger.warning(f"No data found in loki_data")
                    return []

                # self.logger.info(f"Filtering data for Pedestrian ID: {self.pedestrian_id} in scenario: {self.scenario_name}.")

                scenario_to_ped_to_frames = {}
                # for testing, test the first 60 frames
                # frame_count = 0  
                # max_frames = 60
                self.logger.info(f"Extracting all pedestrians with their respective frame(s).")
                for _, row in loki_data.iterrows():
                    frame_name = row['frame_name']
                    scenario_name = row['video_name']
                    # if not scenario_name == 'scenario_026':
                    #     continue
                    ped_id = row['Ped_ID']
                    try:
                        frame_number = int(frame_name.split('_')[-1])  # Ensure we extract numeric frame numbers
                    except ValueError:
                        self.logger.error(f"Invalid frame name format: {frame_name}. Skipping.")
                        continue

                    scenario_to_ped_to_frames.setdefault(scenario_name, {}).setdefault(ped_id, []).append(frame_number)   

                    # frame_count+=1
                    
                    # testing
                    # if frame_count >= max_frames:
                    #     self.logger.info(f"Reached frame limit: {max_frames}. Stopping extraction.")
                    #     # print(scenario_to_ped_to_frames)
                    #     break
                    #   
                # Calculate the total number of pedestrians across all scenarios
                total_peds = sum(len(peds) for peds in scenario_to_ped_to_frames.values())
                self.logger.info(f"Extracted {(total_peds)} pedestrians in all the scenarios.")
                
                # Calculate the total number of frames across all scenarios and pedestrians
                total_frames = sum(len(frames) for peds in scenario_to_ped_to_frames.values() for frames in peds.values())
                self.logger.info(f"Total frames extracted: {total_frames}")

                return scenario_to_ped_to_frames

    def _crop(self, frame: int):
        """
        Crops the map point cloud to the pedestrian's bounding box.

        Args:
            frame (int): Frame number to crop.

        Returns:
            o3d.geometry.PointCloud: The cropped point cloud.

        Notes:
            - Uses the pedestrian's bounding box data from the specified frame.
        """
        self.logger.info(f"Cropping the map to the bounding box of the pedestrian in frame: {frame}.")
        ped_points, bounding_box_o3d = self.pedestrian_data[frame]
        car_points = self.car_data[frame]

        # Ensure the data types are as expected
        assert isinstance(ped_points, np.ndarray) and isinstance(bounding_box_o3d, o3d.geometry.OrientedBoundingBox) and isinstance(car_points, list), "Invalid data types."
        
        if self.use_downsampling:
            self.map_pcd = self.map_pcd.voxel_down_sample(voxel_size=Reconstuction3DConfig.voxel_size)
        
        concat_points = np.concatenate(
                [ped_points, np.concatenate(car_points, axis=0), np.asarray(self.map_pcd.points)], axis=0
                )
        concat_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(concat_points))
        cropped_pcd = concat_pcd.crop(bounding_box_o3d, invert=False)
        
        self.logger.info(f"Finished cropping, {len(cropped_pcd.points)} points for frame: {frame}.")

        return cropped_pcd

   