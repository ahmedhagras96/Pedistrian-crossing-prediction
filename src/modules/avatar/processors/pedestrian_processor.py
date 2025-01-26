import sys

import numpy as np
import open3d as o3d
import pandas as pd

from modules.config.logger import Logger


class PedestrianProcessor:
    """
    Processes pedestrian data, including extraction, averaging, thresholding, and filtering.
    """

    def __init__(self, points_threshold_multiplier: float = 0.5):
        """
        Initialize the PedestrianProcessor with a threshold multiplier.

        Args:
            points_threshold_multiplier (float): Multiplier to set the minimum point threshold
                based on the average. Defaults to 0.5.
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

        self.threshold_multiplier = points_threshold_multiplier
        self.column_names = [
            "labels", "track_id", "stationary", "pos_x", "pos_y", "pos_z",
            "dim_x", "dim_y", "dim_z", "yaw", "vehicle_state",
            "intended_actions", "potential_destination", "additional_info"
        ]
        self.numerical_columns = [
            "pos_x", "pos_y", "pos_z", "dim_x", "dim_y", "dim_z", "yaw"
        ]

    def extract_pedestrian_df(self, labels3d_ndarray: np.ndarray) -> pd.DataFrame:
        """
        Extract pedestrian information from the labels ndarray and return a DataFrame.

        Args:
            labels3d_ndarray (np.ndarray): The labels' data.

        Returns:
            pd.DataFrame: A DataFrame containing pedestrian information.

        Raises:
            SystemExit: If the labels ndarray has an unexpected shape or no pedestrians are found.
        """
        self._validate_labels_array_shape(labels3d_ndarray)

        df_labels3d = self._create_labels_dataframe(labels3d_ndarray)
        self._ensure_numerical_columns(df_labels3d)

        df_pedestrians = self._filter_to_pedestrians(df_labels3d)
        self._check_empty_pedestrians(df_pedestrians)

        return df_pedestrians

    @staticmethod
    def calculate_average_points(pedestrian_pcds: list[o3d.geometry.PointCloud]) -> float:
        """
        Calculate the average number of points across all pedestrian point clouds.

        Args:
            pedestrian_pcds (list[o3d.geometry.PointCloud]): List of pedestrian PCDs.

        Returns:
            float: The average number of points per pedestrian PCD.
        """
        logger = Logger.get_logger("PedestrianProcessor")
        total_points = sum(len(pcd_ped.points) for pcd_ped in pedestrian_pcds)
        avg_points = total_points / len(pedestrian_pcds) if pedestrian_pcds else 0
        logger.info(f"Average number of points per pedestrian PCD: {avg_points:.2f}")
        return avg_points

    def set_min_point_threshold(self, avg_points: float) -> float:
        """
        Set the minimum point threshold based on the average number of points.

        Args:
            avg_points (float): The average number of points per pedestrian PCD.

        Returns:
            float: The minimum point threshold.
        """
        min_threshold = avg_points * self.threshold_multiplier
        self.logger.info(f"Minimum point threshold set to: {min_threshold:.2f}")
        return min_threshold

    @staticmethod
    def filter_pedestrians(
            df_pedestrians: pd.DataFrame,
            pedestrian_pcds: list[o3d.geometry.PointCloud],
            min_threshold: float
    ) -> tuple[pd.DataFrame, list[o3d.geometry.PointCloud]]:
        """
        Filter out pedestrians with point counts below the minimum threshold.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian information.
            pedestrian_pcds (list[o3d.geometry.PointCloud]): List of pedestrian PCDs.
            min_threshold (float): The minimum number of points required.

        Returns:
            tuple[pd.DataFrame, list[o3d.geometry.PointCloud]]:
                A tuple containing the filtered pedestrian DataFrame and the filtered list of PCDs.

        Raises:
            SystemExit: If no pedestrians meet the threshold.
        """
        logger = Logger.get_logger("PedestrianProcessor")
        pedestrian_pcds_filtered = []
        df_pedestrians_filtered = pd.DataFrame(columns=df_pedestrians.columns)

        for (idx, ped_data), pcd_ped in zip(df_pedestrians.iterrows(), pedestrian_pcds):
            point_count = len(pcd_ped.points)
            if point_count >= min_threshold and point_count > 3:
                pedestrian_pcds_filtered.append(pcd_ped)
                df_pedestrians_filtered = pd.concat(
                    [df_pedestrians_filtered, ped_data.to_frame().T],
                    ignore_index=True
                )
            else:
                logger.info(
                    f"Removing pedestrian {ped_data['track_id']} with only {point_count} points."
                )

        if not pedestrian_pcds_filtered:
            logger.error("No pedestrians meet the minimum point threshold.")
            # sys.exit(1)

        logger.info(f"Number of pedestrians after filtering: {len(pedestrian_pcds_filtered)}")
        return df_pedestrians_filtered, pedestrian_pcds_filtered

    def _validate_labels_array_shape(self, labels3d_ndarray: np.ndarray) -> None:
        """
        Validate the shape of the labels ndarray.

        Args:
            labels3d_ndarray (np.ndarray): The labels' data.

        Raises:
            SystemExit: If the labels ndarray does not have the expected dimensions.
        """
        expected_num_columns = len(self.column_names)
        if not (
                labels3d_ndarray.ndim == 2
                and labels3d_ndarray.shape[1] >= expected_num_columns
        ):
            self.logger.error(
                f"labels3d_ndarray has an unexpected shape: {labels3d_ndarray.shape}"
            )
            # sys.exit(1)

    def _create_labels_dataframe(self, labels3d_ndarray: np.ndarray) -> pd.DataFrame:
        """
        Create a DataFrame from the labels ndarray.

        Args:
            labels3d_ndarray (np.ndarray): The labels' data.

        Returns:
            pd.DataFrame: DataFrame containing all label information.
        """
        expected_num_columns = len(self.column_names)
        df_labels3d = pd.DataFrame(
            labels3d_ndarray[:, :expected_num_columns],
            columns=self.column_names
        )
        return df_labels3d

    def _ensure_numerical_columns(self, df_labels3d: pd.DataFrame) -> None:
        """
        Ensure that specified columns in the DataFrame are numeric types.

        Args:
            df_labels3d (pd.DataFrame): DataFrame containing label data.
        """
        for col in self.numerical_columns:
            df_labels3d[col] = pd.to_numeric(df_labels3d[col], errors="coerce")

    def _filter_to_pedestrians(self, df_labels3d: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame to only include rows labeled as 'Pedestrian'.

        Args:
            df_labels3d (pd.DataFrame): DataFrame containing label data.

        Returns:
            pd.DataFrame: DataFrame containing only pedestrian rows.
        """
        df_pedestrians = df_labels3d[
            df_labels3d["labels"] == "Pedestrian"
            ].reset_index(drop=True)
        return df_pedestrians

    def _check_empty_pedestrians(self, df_pedestrians: pd.DataFrame) -> None:
        """
        Check if the pedestrian DataFrame is empty and handle if so.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian information.

        Raises:
            SystemExit: If the pedestrian DataFrame is empty.
        """
        if df_pedestrians.empty:
            self.logger.error("No pedestrian data found in this sample.")
            # raise Exception("No pedestrian data found in this sample.")
