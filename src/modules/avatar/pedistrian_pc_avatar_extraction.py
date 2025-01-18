import pandas as pd
import numpy as np
import open3d as o3d

import os
from glob import glob
from itertools import groupby
from operator import itemgetter
import threading

import json
import torch
import plyfile
from PIL import Image
from torch.utils.data import Dataset

from .utils import *
# from helpers.loki_dataset_handler import LOKIDatasetHandler

class PedestrianProcessor:
    """
    Processes pedestrian data, including extraction, averaging, thresholding, and filtering.
    """

    def __init__(self, points_threshold_multiplier=0.5):
        """
        Initializes the PedestrianProcessor with a threshold multiplier.

        Args:
            threshold_multiplier (float, optional): Multiplier to set the minimum point threshold based on the average
            Defaults to 0.5.
        """
        self.threshold_multiplier = points_threshold_multiplier
        self.column_names = [
            "labels", "track_id", "stationary", "pos_x", "pos_y", "pos_z",
            "dim_x", "dim_y", "dim_z", "yaw", "vehicle_state",
            "intended_actions", "potential_destination", "additional_info"
        ]
        self.numerical_columns = ["pos_x", "pos_y", "pos_z", "dim_x", "dim_y", "dim_z", "yaw"]

    def extract_pedestrian_df(self, labels3d_ndarray):
        """
        Extracts pedestrian information from the labels ndarray and returns a DataFrame.

        Args:
            labels3d_ndarray (numpy.ndarray): The labels' data.

        Returns:
            pd.DataFrame: DataFrame containing pedestrian information.

        Raises:
            SystemExit: If the labels ndarray has an unexpected shape or no pedestrians are found.
        """
        expected_num_columns = len(self.column_names)
        if labels3d_ndarray.ndim == 2 and labels3d_ndarray.shape[1] >= expected_num_columns:
            df_labels3d = pd.DataFrame(labels3d_ndarray[:, :expected_num_columns], columns=self.column_names)
        else:
            print(f"labels3d_ndarray has an unexpected shape: {labels3d_ndarray.shape}")
            exit(1)

        # Ensure numerical columns are correctly typed
        for col in self.numerical_columns:
            df_labels3d[col] = pd.to_numeric(df_labels3d[col], errors='coerce')

        # Filter to include only pedestrians
        df_pedestrians = df_labels3d[df_labels3d['labels'] == 'Pedestrian'].reset_index(drop=True)
        # print("df_pedestrians: ",df_pedestrians)

        #filter pedistrians according to existence in loki.csv file 
        if df_pedestrians.empty:
            print("No pedestrian data found in this sample.")
            exit(1)

        return df_pedestrians

    @staticmethod
    def calculate_average_points(pedestrian_pcds):
        """
        Calculates the average number of points across all pedestrian point clouds.

        Args:
            pedestrian_pcds (list of o3d.geometry.PointCloud): List of pedestrian PCDs.

        Returns:
            float: The average number of points per pedestrian PCD.
        """
        total_points = sum(len(pcd_ped.points) for pcd_ped in pedestrian_pcds)
        avg_points = total_points / len(pedestrian_pcds) if pedestrian_pcds else 0
        print(f"Average number of points per pedestrian PCD: {avg_points:.2f}")
        return avg_points

    def set_min_point_threshold(self, avg_points):
        """
        Sets the minimum point threshold based on the average number of points.

        Args:
            avg_points (float): The average number of points per pedestrian PCD.

        Returns:
            float: The minimum point threshold.
        """
        min_threshold = avg_points * self.threshold_multiplier
        print(f"Minimum point threshold set to: {min_threshold:.2f}")
        return min_threshold

    @staticmethod
    def filter_pedestrians(df_pedestrians, pedestrian_pcds, min_threshold):
        """
        Filters out pedestrians with point counts below the minimum threshold.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian information.
            pedestrian_pcds (list of o3d.geometry.PointCloud): List of pedestrian PCDs.
            min_threshold (float): The minimum number of points required.

        Returns:
            tuple:
                pd.DataFrame: Filtered pedestrian DataFrame.
                list of o3d.geometry.PointCloud: Filtered list of pedestrian PCDs.

        Raises:
            SystemExit: If no pedestrians meet the threshold.
        """
        pedestrian_pcds_filtered = []
        df_pedestrians_filtered = pd.DataFrame(columns=df_pedestrians.columns)

        for (idx, ped_data), pcd_ped in zip(df_pedestrians.iterrows(), pedestrian_pcds):
            point_count = len(pcd_ped.points)
            if point_count >= min_threshold:
                pedestrian_pcds_filtered.append(pcd_ped)
                df_pedestrians_filtered = pd.concat([df_pedestrians_filtered, ped_data.to_frame().T], ignore_index=True)
            else:
                print(f"Removing pedestrian {ped_data['track_id']} with only {point_count} points.")

        if not pedestrian_pcds_filtered:
            print("No pedestrians meet the minimum point threshold.")
            exit(1)

        print(f"Number of pedestrians after filtering: {len(pedestrian_pcds_filtered)}")
        return df_pedestrians_filtered, pedestrian_pcds_filtered
    


class PointCloudProcessor:
    """
    Handles preprocessing of point clouds, including conversion, downsampling, and outlier removal.
    """

    def __init__(self, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
        """
        Initializes the PointCloudProcessor with specified parameters.

        Args:
            voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.02.
            nb_neighbors (int, optional): Number of neighbors for statistical outlier removal. Defaults to 20.
            std_ratio (float, optional): Standard deviation ratio for statistical outlier removal. Defaults to 2.0.
        """
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def preprocess_pcd(self, raw_pcd):
        """
        Preprocesses the raw point cloud by converting, downsampling, and removing outliers.

        Args:
            raw_pcd (Any): The raw point cloud data.

        Returns:
            o3d.geometry.PointCloud: The cleaned and downsampled point cloud.

        Raises:
            SystemExit: If the point cloud cannot be converted.
        """
        # Convert pointcloud to Open3D PointCloud object
        try:
            pcd = convert_from_vertex_to_open3d_pcd(raw_pcd)
        except ValueError as ve:
            print(f"Error converting point cloud: {ve}")
            exit(1)

        print("Downsampling the point cloud...")
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        print("Removing statistical outliers...")
        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        pcd_clean = pcd_down.select_by_index(ind)

        return pcd_clean
    

class BoundingBox:
    """
    Represents a bounding box in 3D space.
    """

    def __init__(self, row):
        """
        Initializes the BoundingBox object from a DataFrame row.

        Args:
            row (pd.Series): A row from the DataFrame containing bounding box info.
        """
        self.row = row
        self.obb = self.create_oriented_bounding_box()

    def create_oriented_bounding_box(self):
        """
        Creates an Oriented Bounding Box (OBB) for the pedestrian.

        Returns:
            o3d.geometry.OrientedBoundingBox or None: The oriented bounding box or None if data is invalid.
        """
        required_fields = ['pos_x', 'pos_y', 'pos_z', 'dim_x', 'dim_y', 'dim_z', 'yaw']
        for field in required_fields:
            if pd.isnull(self.row[field]):
                print(f"Missing field '{field}' in row {self.row.name}. Skipping this bounding box.")
                return None

        # Center position
        center = np.array([self.row['pos_x'], self.row['pos_y'], self.row['pos_z']])

        # Dimensions
        extent = np.array([self.row['dim_x'], self.row['dim_y'], self.row['dim_z']])

        # Yaw angle (rotation around Z-axis)
        yaw = self.row['yaw']

        # Create rotation matrix around Z-axis
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # Create the Oriented Bounding Box
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)

        return obb

class Visualizer:
    """
    Handles visualization of point clouds and bounding boxes using Open3D.
    """

    def __init__(self):
        """
        Initializes the Visualizer.
        """
        self.geometries = []
        self.bounding_boxes = []  # To store bounding box geometries
        self.pedestrian_pcds = []  # To store cropped pedestrian point clouds

    def add_point_cloud(self, pcd):
        """
        Adds a point cloud to the visualization.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud object.
        """
        self.geometries.append(pcd)

    def add_bounding_boxes(self, df_pedestrians, color=[0, 0, 1]):
        """
        Adds pedestrian bounding boxes to the visualization.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian bounding box information.
            color (list, optional): RGB color for the bounding boxes. Defaults to Blue ([0, 0, 1]).
        """
        for idx, row in df_pedestrians.iterrows():
            bbox_obj = BoundingBox(row)
            obb = bbox_obj.obb
            if obb is not None:
                obb.color = color
                self.geometries.append(obb)
                self.bounding_boxes.append(obb)  # Keep track of bounding boxes separately

    def clear_bounding_boxes(self):
        """
        Removes all bounding boxes from the visualization.
        """
        if not self.bounding_boxes:
            print("No bounding boxes to remove.")
            return

        for bbox in self.bounding_boxes:
            if bbox in self.geometries:
                self.geometries.remove(bbox)
        self.bounding_boxes = []  # Clear the list after removal
        print("All bounding boxes have been cleared.")

    def add_coordinate_axes(self, size=5.0, origin=[0, 0, 0]):
        """
        Adds coordinate axes to the visualization.

        Args:
            size (float, optional): Size of the coordinate frame. Defaults to 5.0.
            origin (list, optional): Origin point of the coordinate frame. Defaults to [0, 0, 0].
        """
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        self.geometries.append(axes)

    def extract_pedestrian_pcds(self, pcd, df_pedestrians):
        """
        Extracts pedestrian-specific point clouds by cropping the original point cloud
        based on the bounding boxes.

        Args:
            pcd (o3d.geometry.PointCloud): The original point cloud.
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian bounding box information.

        Returns:
            list of o3d.geometry.PointCloud: List of pedestrian-specific point clouds.
        """
        pedestrian_pcds = []
        for idx, row in df_pedestrians.iterrows():
            bbox = BoundingBox(row).obb
            if bbox is not None:
                # Use the 'crop' method of PointCloud
                cropped_pcd = pcd.crop(bbox)
                pedestrian_pcds.append(cropped_pcd)
        self.pedestrian_pcds = pedestrian_pcds
        return pedestrian_pcds

    def visualize(self, window_name='Visualization', width=1280, height=720):
        """
        Launches the Open3D visualization window with all geometries.

        Args:
            window_name (str, optional): Title of the visualization window. Defaults to 'Visualization'.
            width (int, optional): Width of the window. Defaults to 1280.
            height (int, optional): Height of the window. Defaults to 720.
        """
        o3d.visualization.draw_geometries(
            self.geometries,
            window_name=window_name,
            width=width,
            height=height,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )

    def visualize_pedestrians_only(self, window_name: object = 'Pedestrians Only', width: object = 1280, height: object = 720,
                                   color: object = [1, 0, 0]) -> object:
        """
        Visualizes only the pedestrian point clouds.

        Args:
            window_name (str, optional): Title of the visualization window. Defaults to 'Pedestrians Only'.
            width (int, optional): Width of the window. Defaults to 1280.
            height (int, optional): Height of the window. Defaults to 720.
            color (list, optional): RGB color to assign to all pedestrian point clouds. Defaults to Red ([1, 0, 0]).
        """
        pedestrian_geometries = []
        for pcd in self.pedestrian_pcds:
            # Optionally, assign a color to the pedestrian pcd
            colored_pcd = pcd.paint_uniform_color(color)
            pedestrian_geometries.append(colored_pcd)

        # Add coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        pedestrian_geometries.append(axes)

        # Visualize
        o3d.visualization.draw_geometries(
            pedestrian_geometries,
            window_name=window_name,
            width=width,
            height=height,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )

    def clear_geometries(self):
        """
        Clears all geometries from the visualizer.
        """
        self.geometries = []
        self.bounding_boxes = []
        self.pedestrian_pcds = []
        print("Visualizer geometries cleared.")

class LOKIDatasetHandler:
    """
    Handles loading and accessing samples from the LOKI dataset.
    """

    def __init__(self, root_dir, keys=["pointcloud", "labels_3d"]):
        """
        Initializes the dataset handler.

        Args:
            root_dir (str): Root directory of the LOKI dataset.
            keys (list, optional): Keys to load from the dataset. Defaults to ["pointcloud", "labels_3d"].
        """
        # script_dir = os.path.dirname(os.path.abspath(__file__))

        self.root_dir = root_dir
        # self.root_dir = os.path.join(script_dir, root_dir)
        self.keys = keys
        self.dataset = self._initialize_dataset()

    def _initialize_dataset(self):
        """
        Initializes the LOKIDataset.

        Returns:
            LOKIDataset: Initialized dataset object.
        """
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Provided root_dir '{self.root_dir}' is not a valid directory.")
        return LOKIDataset(root_dir=self.root_dir, keys=self.keys)

    def get_sample(self, index):
        """
        Retrieves a sample from the dataset by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Sample containing pointcloud and labels_3d data.
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Sample index {index} out of range. Dataset size: {len(self.dataset)}.")
        return self.dataset[index]

    def get_sample_by_id(self, scenario_id, frame_id):
        """
        Retrieves a sample from the dataset by index.

        Args:
            scenario_id (int): ID of the scenario to retrieve frame data from.
            frame_id (int): ID of the frame to retrieve data from.

        Returns:
            dict: Sample containing pointcloud and labels_3d data.
        """
        return self.dataset.get_by_id(scenario_id, frame_id)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)


class LOKIDataset(Dataset):
    def __init__(self, root_dir, keys=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the scenarios.
            keys (list of strings): List of keys to specify which data to load.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.keys = (
            keys
            if keys is not None
            else ["odometry", "labels_2d", "labels_3d", "pointcloud", "images", "map"]
        )
        self.transform = transform
        self.scenarios = [
            os.path.join(root_dir, scenario) for scenario in os.listdir(root_dir)
        ]

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scenario_path = self.scenarios[idx]
        sample = {}

        if "odometry" in self.keys:
            sample["odometry"] = self.load_odometry(scenario_path)
        if "labels_2d" in self.keys:
            sample["labels_2d"] = self.load_labels_2d(scenario_path)
        if "labels_3d" in self.keys:
            sample["labels_3d"] = self.load_labels_3d(scenario_path)
        if "pointcloud" in self.keys:
            sample["pointcloud"] = self.load_pointcloud(scenario_path)
        if "images" in self.keys:
            sample["images"] = self.load_images(scenario_path)
        if "map" in self.keys:
            sample["map"] = self.load_map(scenario_path)

        if self.transform and "images" in sample:
            sample["images"] = [self.transform(image) for image in sample["images"]]

        return sample

    def get_by_id(self, scenario_id, frame_id):
        if torch.is_tensor(scenario_id):
            scenario_id = scenario_id.tolist()

        if torch.is_tensor(frame_id):
            frame_id = frame_id.tolist()

        matching_scenario_paths = [
            scenario_path for scenario_path in self.scenarios
            if f'scenario_{scenario_id}' in os.path.basename(scenario_path)
        ]

        scenario_path = matching_scenario_paths[0]

        sample = {}

        if "odometry" in self.keys:
            sample["odometry"] = self.load_odometry(scenario_path, frame_id)
        if "labels_2d" in self.keys:
            sample["labels_2d"] = self.load_labels_2d(scenario_path, frame_id)
        if "labels_3d" in self.keys:
            sample["labels_3d"] = self.load_labels_3d(scenario_path, frame_id)
        if "pointcloud" in self.keys:
            sample["pointcloud"] = self.load_pointcloud(scenario_path, frame_id)
        if "images" in self.keys:
            sample["images"] = self.load_images(scenario_path, frame_id)
        if "map" in self.keys:
            sample["map"] = self.load_map(scenario_path)

        if self.transform and "images" in sample:
            sample["images"] = [self.transform(image) for image in sample["images"]]

        return sample

    @staticmethod
    def load_odometry(scenario_path, frame_id=None):
        # Get all odom files or filter for a specific frame if frame_id is provided
        if frame_id:
            # If frame_id is provided, look for a specific odometry file like 'odom_0024.txt'
            odometry_files = glob(os.path.join(scenario_path, f"odom_{frame_id}.txt"))
        else:
            # Otherwise, load all odometry files
            odometry_files = sorted(glob(os.path.join(scenario_path, "odom_*.txt")))

        # Load odometry data from the selected files
        odometry_data = [
            pd.read_csv(f, header=None, dtype=float).values for f in odometry_files
        ]

        return odometry_data

    @staticmethod
    def load_labels_2d(scenario_path, frame_id=None):
        # Get all label2d files or filter for a specific frame if frame_id is provided
        if frame_id:
            label2d_files = glob(os.path.join(scenario_path, f"label2d_{frame_id}.json"))
        else:
            label2d_files = sorted(glob(os.path.join(scenario_path, "label2d_*.json")))

        # Load label 2d data from the selected files
        labels_2d = [json.load(open(f)) for f in label2d_files]
        return labels_2d

    @staticmethod
    def load_labels_3d(scenario_path, frame_id=None):
        # Get all label3d files or filter for a specific frame if frame_id is provided
        if frame_id:
            label3d_files = glob(os.path.join(scenario_path, f"label3d_{frame_id}.txt"))
        else:
            label3d_files = sorted(glob(os.path.join(scenario_path, "label3d_*.txt")))

        # Load label 3d data from the selected files
        labels_3d = [pd.read_csv(f).values for f in label3d_files]
        return labels_3d

    def load_pointcloud(self, scenario_path, frame_id=None):
        # Get all pointcloud files or filter for a specific frame if frame_id is provided
        if frame_id:
            pointcloud_files = glob(os.path.join(scenario_path, f"pc_{frame_id}.ply"))
        else:
            pointcloud_files = sorted(glob(os.path.join(scenario_path, "pc_*.ply")))

        # Load pointcloud data from the selected files
        pointcloud_data = [self.load_ply(f) for f in pointcloud_files]
        return pointcloud_data

    @staticmethod
    def load_images(scenario_path, frame_id=None):
        # Get all image files or filter for a specific frame if frame_id is provided
        if frame_id:
            image_files = glob(os.path.join(scenario_path, f"image_{frame_id}.png"))
        else:
            image_files = sorted(glob(os.path.join(scenario_path, "image_*.png")))

        # Load images from the selected files
        images = [Image.open(f).convert("RGB") for f in image_files]
        return images

    def load_map(self, scenario_path):
        map_file = os.path.join(scenario_path, "map.ply")
        map_data = self.load_ply(map_file)
        return map_data

    def load_ply(self, file_path):
        plydata = plyfile.PlyData.read(file_path)
        return np.array([list(vertex) for vertex in plydata.elements[0]])

class PedestrianProcessingPipeline:
    """
    Encapsulates the workflow for processing and visualizing pedestrian point clouds.
    """

    def __init__(self, root_dir, csv_path, save_dir="saved_pedestrians", threshold_multiplier=0.5):
        """
        Initializes the processing pipeline.

        Args:
            root_dir (str): Root directory of the dataset.
            csv_path (str): Path to the CSV file containing pedestrian data.
            threshold_multiplier (float, optional): Multiplier for setting the minimum point threshold. Defaults to 0.5.
            save_dir (str, optional): Directory to save pedestrian point clouds. Defaults to "saved_pedestrians".
        """
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.threshold_multiplier = threshold_multiplier
        self.save_dir = save_dir

        # Initialize Handlers and Processors
        self.dataset_handler = LOKIDatasetHandler(root_dir=self.root_dir, keys=["pointcloud", "labels_3d"])
        self.pointcloud_processor = PointCloudProcessor()
        self.pedestrian_processor = PedestrianProcessor(points_threshold_multiplier=self.threshold_multiplier)
        self.visualizer = Visualizer()

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory for saving pedestrian PCDs: {self.save_dir}")

    def load_scenario_frame_ids(self):
        """
        Loads unique scenario and frame IDs from the CSV file.

        Returns:
            tuple: Tuple containing arrays of scenario_ids and frame_ids.
        """
        print(f"Retrieving scenario & frame IDs with pedestrians from file {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        target_columns = df[['video_name', 'frame_name']]

        # Extract scenario IDs by removing the 'scenario_' prefix
        scenario_ids = target_columns['video_name'].apply(lambda x: x.split('_', 1)[1]).unique()
        # Extract frame IDs by removing the 'image_' prefix
        frame_ids = target_columns['frame_name'].apply(lambda x: x.split('_', 1)[1]).unique()

        print(f"Found {len(scenario_ids)} unique scenarios and {len(frame_ids)} unique frames.")
        return scenario_ids, frame_ids

    def verify_scenarios(self, all_scenario_ids):
        """
        Verifies the existence of scenario directories.

        Args:
            all_scenario_ids (array-like): Array of scenario IDs to verify.

        Returns:
            list: List of valid scenario IDs that exist.
        """
        missing_scenarios = []
        valid_scenarios = []

        for scenario_id in all_scenario_ids:
            if self._scenario_exists(scenario_id):
                valid_scenarios.append(scenario_id)
                # print(f"Scenario {scenario_id} exists.")
            else:
                missing_scenarios.append(int(scenario_id))  # Convert to int for processing ranges
                # print(f"Scenario {scenario_id} is missing.")

        if missing_scenarios:
            missing_ranges = self._group_consecutive_ids(sorted(missing_scenarios))
            self._print_missing_scenarios(missing_ranges)
        else:
            print("All scenarios exist.")

        return valid_scenarios

    def verify_frames(self, valid_scenario_ids, all_frame_ids):
        """
        Verifies the existence of frame files within valid scenarios.

        Args:
            valid_scenario_ids (list): List of valid scenario IDs.
            all_frame_ids (list): List of all frame IDs to verify.

        Returns:
            list: List of valid frame IDs that exist across all valid scenarios.
        """
        missing_frames = {}
        valid_frame_ids = set(all_frame_ids)  # Initialize with all frames

        for scenario_id in valid_scenario_ids:
            scenario_dir = os.path.join(self.root_dir, f'scenario_{scenario_id}')
            existing_frames = self._get_existing_frames(scenario_dir, all_frame_ids)
            missing = set(all_frame_ids) - existing_frames

            if missing:
                missing_frames[scenario_id] = sorted(missing)
                valid_frame_ids &= existing_frames  # Intersection to ensure frame exists across all scenarios
                print(f"Scenario {scenario_id}: Missing frames {sorted(missing)}")
            else:
                valid_frame_ids &= existing_frames
                print(f"Scenario {scenario_id}: All frames exist.")

        if missing_frames:
            self._print_missing_frames(missing_frames)
        else:
            print("All frames exist for the valid scenarios.")

        return sorted(valid_frame_ids)

    def process_all_frames_and_crop_pedestrians(self, valid_scenario_ids, valid_frame_ids):
        """
        Processes all valid scenarios and frames, crops pedestrian point clouds,
        and saves them asynchronously.

        Args:
            valid_scenario_ids (list): List of valid scenario IDs.
            valid_frame_ids (list): List of valid frame IDs.
        """
        all_pedestrians_filtered = []
        for scenario_id in valid_scenario_ids:
            for frame_id in valid_frame_ids:
                print(f"\nProcessing Scenario: {scenario_id}, Frame: {frame_id}")

                # Retrieve sample
                raw_pcd, labels3d_ndarray = self.get_pcd_and_labels(scenario_id, frame_id)
                if raw_pcd is None or labels3d_ndarray is None:
                    continue

                # Preprocess Point Cloud
                cleaned_pcd = self.pointcloud_processor.preprocess_pcd(raw_pcd)
                print(f"Preprocessed point cloud for Scenario: {scenario_id}, Frame: {frame_id}")

                # Extract Pedestrian DataFrame 
                df_pedestrians = self.pedestrian_processor.extract_pedestrian_df(labels3d_ndarray)

                #filter the return dataframe according if the pedistrian exist in loki.csv or not
                df_loki = pd.read_csv(self.csv_path)  # DataFrame for loki.csv
                filtered_pedestrians = df_pedestrians[df_pedestrians['track_id'].isin(df_loki['Ped_ID'])]

                print(f"Extracted {len(filtered_pedestrians)} pedestrians in Scenario: {scenario_id}, Frame: {frame_id}")

                if df_pedestrians.empty:
                    print(f"No pedestrians found in Scenario: {scenario_id}, Frame: {frame_id}.")
                    continue
                    
                # Extract Pedestrian Point Clouds
                pedestrian_pcds = self.visualizer.extract_pedestrian_pcds(cleaned_pcd, filtered_pedestrians)
                print(f"Extracted {len(pedestrian_pcds)} pedestrian point clouds.")

                if not pedestrian_pcds:
                    print("No pedestrian point clouds extracted.")
                    continue

                # Calculate Average Number of Points and filter low-count pedestrians
                avg_points = self.pedestrian_processor.calculate_average_points(pedestrian_pcds)

                # Set Minimum Point Threshold
                min_threshold = self.pedestrian_processor.set_min_point_threshold(avg_points)

                # Filter Pedestrians Based on Threshold
                df_pedestrians_filtered, pedestrian_pcds_filtered = self.pedestrian_processor.filter_pedestrians(
                    df_pedestrians, pedestrian_pcds, min_threshold
                )

                # Recenter and prepare pedestrian PCDs
                pedestrian_pcd_dict = self._prepare_pedestrian_pcds(
                    scenario_id, frame_id, df_pedestrians_filtered, pedestrian_pcds_filtered
                )
                print(f"Prepared pedestrian PCD dictionary for Scenario: {scenario_id}, Frame: {frame_id}")

                if df_pedestrians.empty == False:
                    df_pedestrians_filtered['scenario_id'] = scenario_id
                    df_pedestrians_filtered['frame_id'] = frame_id
                    df_ped_filt_cols = df_pedestrians_filtered[['track_id', 'scenario_id', 'frame_id', 'intended_actions']]
                    all_pedestrians_filtered.append(df_ped_filt_cols)
                    
                # Save pedestrian PCDs asynchronously
                save_thread = threading.Thread(target=self.save_pedestrian_pcds, args=(pedestrian_pcd_dict,))
                save_thread.start()
                print(f"Started saving pedestrian PCDs for Scenario: {scenario_id}, Frame: {frame_id} asynchronously.")

        combined_df = pd.concat(all_pedestrians_filtered, ignore_index=True) if all_pedestrians_filtered else pd.DataFrame()
        print("\nProcessing completed.")
        return combined_df

    def get_pcd_and_labels(self, scenario_id, frame_id):
        # Retrieve sample
        try:
            sample = self.dataset_handler.get_sample_by_id(scenario_id, frame_id)
            print(f"Retrieved sample for Scenario: {scenario_id}, Frame: {frame_id}")
        except Exception as e:
            print(f"Error retrieving sample for Scenario: {scenario_id}, Frame: {frame_id}: {e}")
            return None, None

        if any(not v for v in sample.values()):
            print(f"Skipping Scenario: {scenario_id}, Frame: {frame_id} as no values were found.")
            return None, None

        # Extract Point Cloud and Labels
        raw_pcd = sample.get("pointcloud", [])[0]
        labels3d_ndarray = sample.get("labels_3d", [])[0]

        return raw_pcd, labels3d_ndarray

    def save_pedestrian_pcds(self, pcd_dict):
        """
        Saves the pedestrian point clouds in the provided dictionary as .ply files asynchronously.

        Args:
            pcd_dict (dict): Dictionary where keys are a combination of scenario_id, frame_id, and pedestrian_id,
                            and values are the cropped pedestrian point clouds.
        """
        for key, pcd in pcd_dict.items():
            # Construct the filename
            filename = f"{key}.ply"
            filepath = os.path.join(self.save_dir, filename)

            # Save the point cloud to a .ply file
            try:
                o3d.io.write_point_cloud(filepath, pcd)
                print(f"Saved pedestrian PCD: {filepath}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")

        print("All pedestrian point clouds for the current frame have been saved.")

    # ----------------------- Helper Methods -----------------------

    def _scenario_exists(self, scenario_id):
        """
        Checks if a scenario directory exists.

        Args:
            scenario_id (str): Scenario ID to check.

        Returns:
            bool: True if scenario exists, False otherwise.
        """
        scenario_dir = os.path.join(self.root_dir, f'scenario_{scenario_id}')
        return os.path.isdir(scenario_dir)

    @staticmethod
    def _group_consecutive_ids(id_list):
        """
        Groups consecutive IDs into ranges.

        Args:
            id_list (list): Sorted list of integer IDs.

        Returns:
            list: List of grouped ID ranges as strings.
        """
        ranges = []
        for k, g in groupby(enumerate(id_list), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) > 1:
                ranges.append(f"{str(group[0]).zfill(3)}-{str(group[-1]).zfill(3)}")
            else:
                ranges.append(str(group[0]).zfill(3))
        return ranges

    def _print_missing_scenarios(self, missing_ranges):
        """
        Prints the missing scenarios in grouped ranges.

        Args:
            missing_ranges (list): List of missing scenario ID ranges.
        """
        if len(missing_ranges) == 1 and '-' in missing_ranges[0]:
            print(f"Skipping the following scenarios because they do not exist in {self.root_dir}:")
            print(f"{missing_ranges[0]}")
        else:
            print(f"Skipping the following scenarios because they do not exist in {self.root_dir}:")
            print(", ".join(missing_ranges))

    @staticmethod
    def _get_existing_frames(scenario_dir, frame_ids):
        """
        Retrieves existing frame IDs within a scenario directory.

        Args:
            scenario_dir (str): Path to the scenario directory.
            frame_ids (list): List of frame IDs to check.

        Returns:
            set: Set of existing frame IDs.
        """
        existing_frames = set()
        for frame_id in frame_ids:
            # Check if any file with the pattern *_XXXX (frame_id) exists
            frame_files = glob(os.path.join(scenario_dir, f"*_{frame_id}.*"))
            if frame_files:
                existing_frames.add(frame_id)
        return existing_frames

    def _print_missing_frames(self, missing_frames):
        """
        Prints the missing frames for each scenario.

        Args:
            missing_frames (dict): Dictionary mapping scenario IDs to lists of missing frame IDs.
        """
        for scenario_id, frames in missing_frames.items():
            grouped_frames = self._group_consecutive_ids([int(f) for f in frames])
            print(f"Skipping the following frames in scenario {scenario_id} because they do not exist:")
            print(", ".join(grouped_frames))

    @staticmethod
    def _prepare_pedestrian_pcds(scenario_id, frame_id, df_pedestrians_filtered, pedestrian_pcds_filtered):
        """
        Prepares the pedestrian PCDs by recentering and creating a dictionary.

        Args:
            scenario_id (str): Scenario ID.
            frame_id (str): Frame ID.
            df_pedestrians_filtered (pd.DataFrame): Filtered pedestrian DataFrame.
            pedestrian_pcds_filtered (list): List of filtered pedestrian point clouds.

        Returns:
            dict: Dictionary with unique keys and pedestrian PCDs.
        """
        pedestrian_pcd_dict = {}
        # for idx, pcd in enumerate(pedestrian_pcds_filtered):
        for idx, pcd in zip(df_pedestrians_filtered['track_id'],pedestrian_pcds_filtered):
            # pedestrian_id = df_pedestrians_filtered.iloc[idx].get("Ped_ID", idx)

            # Recenter the point cloud (optional based on requirements)
            points = np.asarray(pcd.points)
            centroid = np.mean(points, axis=0)
            points_recentered = points - centroid
            pcd.points = o3d.utility.Vector3dVector(points_recentered)

            # Create a unique key for each pedestrian (scenario_id_frame_id_pedestrian_id)
            pedestrian_key = f"{scenario_id}_{frame_id}_ped_{idx}"
            pedestrian_pcd_dict[pedestrian_key] = pcd

        return pedestrian_pcd_dict


# Define the paths directly
# root_dir = '../LOKI'
# root_dir = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\LOKI'
# csv_path = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\LOKI\\b_loki.csv'
# threshold_multiplier = 0.5
# output_csv = '../LOKI/pedestrian_pointclouds.csv'
# save_directory = 'D:\AUC_RA\pedestrian behavior prediction using pointcloud with images\Pedistrian-crossing-prediction\processed_scenarios\saved_pedestrians'
# # Create the pipeline instance
# pipeline = PedestrianProcessingPipeline(
#     root_dir=root_dir,
#     csv_path=csv_path,
#     save_dir=save_directory,
#     threshold_multiplier=threshold_multiplier
# )

# try:

#   # Load scenario and frame IDs
#   scenario_ids, frame_ids = pipeline.load_scenario_frame_ids()

#   # Verify scenarios
#   valid_scenario_ids = pipeline.verify_scenarios(scenario_ids)

#   # Verify frames
#   valid_frame_ids = pipeline.verify_frames(valid_scenario_ids, frame_ids)

#   # Process all valid frames & save pcds
#   pipeline.process_all_frames_and_crop_pedestrians(valid_scenario_ids, valid_frame_ids)

# except KeyboardInterrupt:
#   print("\nOperation cancelled by user. Exiting gracefully...")
