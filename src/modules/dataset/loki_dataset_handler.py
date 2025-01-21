import os
from typing import List, Dict, Any

from modules.dataset.loki_dataset import LOKIDataset
from modules.config.logger import Logger


class LOKIDatasetHandler:
    """
    Handles loading and accessing samples from the LOKI dataset.
    """

    def __init__(
            self,
            root_dir: str,
            keys: List[str] = None
    ) -> None:
        """
        Initializes the dataset handler.

        Args:
            root_dir (str): Root directory of the LOKI dataset.
            keys (List[str], optional): Keys to load from the dataset. Defaults to ["pointcloud", "labels_3d"].

        Raises:
            ValueError: If the root directory is invalid.
        """
        # Replace 'Logger' with your actual logger class import or definition
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        if keys is None:
            keys = ["pointcloud", "labels_3d"]

        self.root_dir = root_dir
        self.keys = keys
        self.dataset = self._initialize_dataset()

        self.logger.info(f"{self.__class__.__name__} initialized successfully with "
                         f"root_dir='{self.root_dir}' and keys={self.keys}.")

    def _initialize_dataset(self):
        """
        Validates the root directory and initializes the LOKIDataset.

        Returns:
            LOKIDataset: The initialized dataset object.

        Raises:
            ValueError: If the root_dir is not a valid directory.
        """
        self._validate_root_directory(self.root_dir)
        dataset = LOKIDataset(root_dir=self.root_dir, keys=self.keys)
        self.logger.info("LOKIDataset initialized successfully.")
        return dataset

    def _validate_root_directory(self, directory: str) -> None:
        """
        Validates that the specified directory exists.

        Args:
            directory (str): The directory to validate.

        Raises:
            ValueError: If the directory does not exist.
        """
        if not os.path.isdir(directory):
            error_message = f"Provided root_dir '{directory}' is not a valid directory."
            self.logger.error(error_message)
            raise ValueError(error_message)

    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset by its integer index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the sample's data (e.g., pointcloud, labels_3d).

        Raises:
            IndexError: If the index is out of the dataset range.
        """
        if index < 0 or index >= len(self.dataset):
            error_message = (f"Sample index {index} is out of range. "
                             f"Dataset size: {len(self.dataset)}.")
            self.logger.error(error_message)
            raise IndexError(error_message)

        self.logger.info(f"Retrieving sample at index {index}.")
        return self.dataset[index]

    def get_sample_by_id(self, scenario_id: int, frame_id: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset by scenario ID and frame ID.

        Args:
            scenario_id (int): ID of the scenario to retrieve frame data from.
            frame_id (int): ID of the frame to retrieve data from.

        Returns:
            Dict[str, Any]: A dictionary containing the sample's data.

        Raises:
            KeyError: If the specified scenario_id or frame_id is invalid in the dataset.
        """
        self.logger.info(f"Retrieving sample by scenario_id={scenario_id}, frame_id={frame_id}.")
        try:
            sample = self.dataset.get_by_id(scenario_id, frame_id)
        except KeyError as ke:
            self.logger.error(f"Invalid scenario_id={scenario_id} or frame_id={frame_id}: {ke}")
            raise
        return sample

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)
