import json
import os
from glob import glob
from typing import List, Dict, Any, Optional, Callable, Union

import numpy as np
import pandas as pd
import plyfile
import torch
from PIL import Image
from torch.utils.data import Dataset

from modules.config.logger import Logger


class LOKIDataset(Dataset):
    """
    A PyTorch Dataset for loading LOKI scenario data, including odometry, labels, pointclouds, images, and maps.
    """

    def __init__(
            self,
            root_dir: str,
            keys: Optional[List[str]] = None,
            transform: Optional[Callable] = None
    ) -> None:
        """
        Initializes the LOKIDataset.

        Args:
            root_dir (str): Path to the directory containing all scenarios.
            keys (List[str], optional): List of dataset keys specifying which data to load.
                Defaults to ["odometry", "labels_2d", "labels_3d", "pointcloud", "images", "map"].
            transform (Callable, optional): An optional transform function or callable to be applied
                to images once they are loaded.
        """
        # Replace 'Logger' with your actual logger class import or definition
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        if keys is None:
            keys = ["odometry", "labels_2d", "labels_3d", "pointcloud", "images", "map"]

        self.root_dir = root_dir
        self.keys = keys
        self.transform = transform

        # Gather all scenario subdirectories in the root directory
        self.scenarios = self._get_scenario_paths(root_dir)

        self.logger.info(
            f"{self.__class__.__name__} initialized with root_dir='{root_dir}', "
            f"keys={keys}, and {len(self.scenarios)} scenario(s) found."
        )

    def _get_scenario_paths(self, directory: str) -> List[str]:
        """
        Retrieves the list of scenario paths from the specified directory.

        Args:
            directory (str): Path to the root directory containing scenario subdirectories.

        Returns:
            List[str]: A list of full paths to scenario subdirectories.
        """
        if not os.path.isdir(directory):
            error_msg = f"The provided directory '{directory}' is invalid or does not exist."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        paths = [
            os.path.join(directory, scenario)
            for scenario in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, scenario))
        ]
        self.logger.info(f"Found {len(paths)} scenario(s) in '{directory}'.")
        return paths

    def __len__(self) -> int:
        """
        Returns the total number of scenarios in the dataset.

        Returns:
            int: The number of scenario directories.
        """
        return len(self.scenarios)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset by the scenario index.

        Args:
            idx (int or torch.Tensor): Index specifying which scenario to load.

        Returns:
            Dict[str, Any]: A dictionary where keys are the data types (e.g., 'odometry', 'labels_3d')
            and values are the loaded data.

        Raises:
            IndexError: If the provided index is out of the range of scenarios.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < 0 or idx >= len(self.scenarios):
            error_msg = f"Index {idx} is out of range. Dataset size: {len(self.scenarios)}."
            self.logger.error(error_msg)
            raise IndexError(error_msg)

        scenario_path = self.scenarios[idx]
        self.logger.info(f"Loading scenario at index {idx}: {scenario_path}")

        # Collect data specified by keys
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

        # Optionally apply transform to images
        if self.transform and "images" in sample:
            self.logger.info("Applying transform to loaded images.")
            sample["images"] = [self.transform(image) for image in sample["images"]]

        return sample

    def get_by_id(self, scenario_id: int, frame_id: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset by scenario ID and frame ID.

        Args:
            scenario_id (int): Identifier for the scenario (used in directory naming).
            frame_id (int): Frame number or identifier for files in the scenario directory.

        Returns:
            Dict[str, Any]: A dictionary where keys are the data types and values are the loaded data.
        """
        self.logger.info(f"Searching for scenario with ID={scenario_id} and frame ID={frame_id}.")
        matching_scenario_paths = [
            scenario_path
            for scenario_path in self.scenarios
            if f"scenario_{scenario_id}" in os.path.basename(scenario_path)
        ]

        if not matching_scenario_paths:
            error_msg = f"No scenario found for scenario_id={scenario_id}."
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        scenario_path = matching_scenario_paths[0]
        self.logger.info(f"Found matching scenario directory: {scenario_path}")

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

        # Optionally apply transform to images
        if self.transform and "images" in sample:
            self.logger.info("Applying transform to loaded images for scenario/frame ID query.")
            sample["images"] = [self.transform(image) for image in sample["images"]]

        return sample

    @staticmethod
    def load_odometry(scenario_path: str, frame_id: Optional[int] = None) -> List[np.ndarray]:
        """
        Loads odometry data from text files. If a frame_id is provided, it attempts to load
        odometry from the file named 'odom_<frame_id>.txt'. Otherwise, it loads all odom_*.txt files.

        Args:
            scenario_path (str): Path to the scenario directory.
            frame_id (int, optional): Specific frame ID to load. Defaults to None.

        Returns:
            List[np.ndarray]: A list of numpy arrays containing odometry data for each file found.
        """
        if frame_id is not None:
            odometry_files = glob(os.path.join(scenario_path, f"odom_{frame_id}.txt"))
        else:
            odometry_files = sorted(glob(os.path.join(scenario_path, "odom_*.txt")))

        return [
            pd.read_csv(f, header=None, dtype=float).values
            for f in odometry_files
        ]

    @staticmethod
    def load_labels_2d(scenario_path: str, frame_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Loads 2D label data from JSON files. If a frame_id is provided, it attempts to load
        from the file named 'label2d_<frame_id>.json'. Otherwise, it loads all label2d_*.json files.

        Args:
            scenario_path (str): Path to the scenario directory.
            frame_id (int, optional): Specific frame ID to load. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing label data.
        """
        if frame_id is not None:
            label2d_files = glob(os.path.join(scenario_path, f"label2d_{frame_id}.json"))
        else:
            label2d_files = sorted(glob(os.path.join(scenario_path, "label2d_*.json")))

        return [json.load(open(f)) for f in label2d_files]

    @staticmethod
    def load_labels_3d(scenario_path: str, frame_id: Optional[int] = None) -> List[np.ndarray]:
        """
        Loads 3D label data from text files. If a frame_id is provided, it attempts to load
        from the file named 'label3d_<frame_id>.txt'. Otherwise, it loads all label3d_*.txt files.

        Args:
            scenario_path (str): Path to the scenario directory.
            frame_id (int, optional): Specific frame ID to load. Defaults to None.

        Returns:
            List[np.ndarray]: A list of numpy arrays containing 3D label data.
        """
        if frame_id is not None:
            label3d_files = glob(os.path.join(scenario_path, f"label3d_{frame_id}.txt"))
        else:
            label3d_files = sorted(glob(os.path.join(scenario_path, "label3d_*.txt")))

        return [pd.read_csv(f).values for f in label3d_files]

    def load_pointcloud(self, scenario_path: str, frame_id: Optional[int] = None) -> List[np.ndarray]:
        """
        Loads pointcloud data from PLY files. If a frame_id is provided, it attempts to load
        from the file named 'pc_<frame_id>.ply'. Otherwise, it loads all pc_*.ply files.

        Args:
            scenario_path (str): Path to the scenario directory.
            frame_id (int, optional): Specific frame ID to load. Defaults to None.

        Returns:
            List[np.ndarray]: A list of numpy arrays representing pointcloud data.
        """
        if frame_id is not None:
            pointcloud_files = glob(os.path.join(scenario_path, f"pc_{frame_id}.ply"))
        else:
            pointcloud_files = sorted(glob(os.path.join(scenario_path, "pc_*.ply")))

        return [self.load_ply(f) for f in pointcloud_files]

    @staticmethod
    def load_images(scenario_path: str, frame_id: Optional[int] = None) -> List[Image.Image]:
        """
        Loads image data from PNG files. If a frame_id is provided, it attempts to load
        from the file named 'image_<frame_id>.png'. Otherwise, it loads all image_*.png files.

        Args:
            scenario_path (str): Path to the scenario directory.
            frame_id (int, optional): Specific frame ID to load. Defaults to None.

        Returns:
            List[Image.Image]: A list of PIL Images.
        """
        if frame_id is not None:
            image_files = glob(os.path.join(scenario_path, f"image_{frame_id}.png"))
        else:
            image_files = sorted(glob(os.path.join(scenario_path, "image_*.png")))

        return [Image.open(f).convert("RGB") for f in image_files]

    def load_map(self, scenario_path: str) -> np.ndarray:
        """
        Loads a map from a PLY file named 'map.ply' located in the scenario path.

        Args:
            scenario_path (str): Path to the scenario directory.

        Returns:
            np.ndarray: A numpy array representing the loaded map pointcloud.
        """
        map_file = os.path.join(scenario_path, "map.ply")
        return self.load_ply(map_file)

    def load_ply(self, file_path: str) -> np.ndarray:
        """
        Loads point data from a PLY file and returns it as a numpy array.

        Args:
            file_path (str): Path to the PLY file.

        Returns:
            np.ndarray: A numpy array containing the loaded point data.
        """
        plydata = plyfile.PlyData.read(file_path)
        self.logger.info(f"Loaded PLY file: {file_path}")
        return np.array([list(vertex) for vertex in plydata.elements[0]])
