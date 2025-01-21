import pandas as pd

from modules.avatar.extraction.pedestrian_manager import PedestrianPCDManager
from modules.avatar.extraction.scenario_validator import ScenarioValidator
from modules.avatar.processors.pedestrian_processor import PedestrianProcessor
from modules.avatar.processors.pointcloud_processor import PointCloudProcessor
from modules.avatar.utilities.visualizer import Visualizer
from modules.config.logger import Logger
from modules.dataset.loki_dataset_handler import LOKIDatasetHandler


class AvatarExtractionPipeline:
    """
    Orchestrates the workflow for processing and visualizing pedestrian point clouds.
    """

    def __init__(
            self,
            root_dir: str,
            csv_path: str,
            save_dir: str = "saved_pedestrians",
            threshold_multiplier: float = 0.5
    ) -> None:
        """
        Initializes the processing pipeline.

        Args:
            root_dir (str): Root directory of the dataset.
            csv_path (str): Path to the CSV file containing pedestrian data.
            threshold_multiplier (float, optional): Multiplier for setting the minimum point threshold.
                Defaults to 0.5.
            save_dir (str, optional): Directory to save pedestrian point clouds. Defaults to "saved_pedestrians".
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        self.root_dir = root_dir
        self.csv_path = csv_path
        self.save_dir = save_dir
        self.threshold_multiplier = threshold_multiplier

        # Initialize supporting classes
        self.dataset_handler = LOKIDatasetHandler(
            root_dir=self.root_dir,
            keys=["pointcloud", "labels_3d"]
        )
        self.pointcloud_processor = PointCloudProcessor()
        self.pedestrian_processor = PedestrianProcessor(points_threshold_multiplier=self.threshold_multiplier)
        self.visualizer = Visualizer()

        # Our new helper classes
        self.scenario_validator = ScenarioValidator(root_dir)
        self.pcd_manager = PedestrianPCDManager(self.save_dir)

        self.logger.info(f"Initialized {self.__class__.__name__} with CSV='{csv_path}'")

    def load_scenario_frame_ids(self) -> tuple:
        """
        Loads unique scenario and frame IDs from the CSV file.

        Returns:
            tuple: A tuple of (scenario_ids, frame_ids).
        """
        self.logger.info(f"Retrieving scenario & frame IDs from CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        target_columns = df[['video_name', 'frame_name']]

        scenario_ids = target_columns['video_name'].apply(lambda x: x.split('_', 1)[1]).unique()
        frame_ids = target_columns['frame_name'].apply(lambda x: x.split('_', 1)[1]).unique()

        self.logger.info(f"Found {len(scenario_ids)} unique scenarios and {len(frame_ids)} unique frames.")
        return scenario_ids, frame_ids

    def validate_scenarios_and_frames(self, scenario_ids, frame_ids):
        """
        Validates the provided scenario and frame IDs.

        Args:
            scenario_ids (array-like): Collection of scenario IDs from CSV.
            frame_ids (array-like): Collection of frame IDs from CSV.

        Returns:
            tuple: A tuple containing (valid_scenario_ids, valid_frame_ids).
        """
        self.logger.info("Validating scenarios...")
        valid_scenarios = self.scenario_validator.verify_scenarios(scenario_ids)

        self.logger.info("Validating frames for each valid scenario...")
        valid_frames = self.scenario_validator.verify_frames(valid_scenarios, frame_ids)

        return valid_scenarios, valid_frames

    def process_all_frames_and_crop_pedestrians(self, valid_scenario_ids, valid_frame_ids) -> pd.DataFrame:
        """
        Processes all valid scenarios/frames, crops pedestrian point clouds, filters them,
        and saves them asynchronously. Returns a DataFrame of filtered pedestrian metadata.

        Args:
            valid_scenario_ids (list): List of valid scenario IDs.
            valid_frame_ids (list): List of valid frame IDs.

        Returns:
            pd.DataFrame: A DataFrame containing metadata about filtered pedestrians across all frames.
        """
        all_pedestrians_filtered = []

        for scenario_id in valid_scenario_ids:
            for frame_id in valid_frame_ids:
                self.logger.info(f"Processing Scenario: {scenario_id}, Frame: {frame_id}")

                # Retrieve sample
                raw_pcd, labels3d_ndarray = self._get_pcd_and_labels(scenario_id, frame_id)
                if raw_pcd is None or labels3d_ndarray is None:
                    continue

                # Preprocess Point Cloud
                cleaned_pcd = self.pointcloud_processor.preprocess_pcd(raw_pcd)
                self.logger.info(f"Preprocessed point cloud for Scenario: {scenario_id}, Frame: {frame_id}")

                # Extract Pedestrian DataFrame
                df_pedestrians = self.pedestrian_processor.extract_pedestrian_df(labels3d_ndarray)

                # Filter pedestrians using the CSV
                df_loki = pd.read_csv(self.csv_path)
                filtered_pedestrians = df_pedestrians[df_pedestrians['track_id'].isin(df_loki['Ped_ID'])]

                self.logger.info(f"Extracted {len(filtered_pedestrians)} pedestrian(s) in "
                                 f"Scenario: {scenario_id}, Frame: {frame_id}")

                if df_pedestrians.empty:
                    self.logger.warning(f"No pedestrians found in Scenario: {scenario_id}, Frame: {frame_id}")
                    continue

                # Extract Pedestrian Point Clouds
                pedestrian_pcds = self.visualizer.extract_pedestrian_pcds(cleaned_pcd, filtered_pedestrians)
                self.logger.info(f"Extracted {len(pedestrian_pcds)} pedestrian point cloud(s).")

                if not pedestrian_pcds:
                    self.logger.warning("No pedestrian point clouds extracted.")
                    continue

                # Calculate Average Number of Points and filter
                avg_points = self.pedestrian_processor.calculate_average_points(pedestrian_pcds)
                min_threshold = self.pedestrian_processor.set_min_point_threshold(avg_points)

                df_pedestrians_filtered, pedestrian_pcds_filtered = self.pedestrian_processor.filter_pedestrians(
                    df_pedestrians, pedestrian_pcds, min_threshold
                )

                # Prepare pedestrian PCDs
                pedestrian_pcd_dict = self.pcd_manager.prepare_pedestrian_pcds(
                    scenario_id,
                    frame_id,
                    df_pedestrians_filtered,
                    pedestrian_pcds_filtered
                )

                # Optionally track metadata
                if not df_pedestrians.empty:
                    df_pedestrians_filtered['scenario_id'] = scenario_id
                    df_pedestrians_filtered['frame_id'] = frame_id
                    df_ped_filtered_cols = df_pedestrians_filtered[
                        ['track_id', 'scenario_id', 'frame_id', 'intended_actions']]
                    all_pedestrians_filtered.append(df_ped_filtered_cols)

                # Save PCDs asynchronously
                self.pcd_manager.save_pedestrian_pcds(pedestrian_pcd_dict)
                self.logger.info(f"Saving pedestrian PCDs for Scenario: {scenario_id}, Frame: {frame_id}")

        combined_df = pd.concat(all_pedestrians_filtered,
                                ignore_index=True) if all_pedestrians_filtered else pd.DataFrame()
        self.logger.info("Processing completed.")
        return combined_df

    def _get_pcd_and_labels(self, scenario_id: str, frame_id: str):
        """
        Retrieves the point cloud and labels from the dataset.

        Args:
            scenario_id (str): Scenario ID.
            frame_id (str): Frame ID.

        Returns:
            tuple: (raw_pcd, labels3d_ndarray) if available, otherwise (None, None).
        """
        try:
            sample = self.dataset_handler.get_sample_by_id(scenario_id, frame_id)
            self.logger.info(f"Retrieved sample for Scenario: {scenario_id}, Frame: {frame_id}")
        except Exception as e:
            self.logger.error(f"Error retrieving sample for Scenario: {scenario_id}, Frame: {frame_id}: {e}")
            return None, None

        if any(not v for v in sample.values()):
            self.logger.warning(f"Skipping Scenario: {scenario_id}, Frame: {frame_id} as no values were found.")
            return None, None

        raw_pcd = sample.get("pointcloud", [])[0]
        labels3d_ndarray = sample.get("labels_3d", [])[0]
        return raw_pcd, labels3d_ndarray
