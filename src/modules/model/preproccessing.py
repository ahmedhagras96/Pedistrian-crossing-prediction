import time
from pathlib import Path

import pandas as pd

from modules.avatar.extraction.avatar_extraction_pipeline import AvatarExtractionPipeline
from modules.config.logger import Logger
from modules.config.paths_loader import PATHS
from modules.features.pedestrian_features.extractors.pedestrian_feature_extractor import PedestrianFeatureExtractor
from modules.reconstruction.aligners.pedestrian_map_aligner import PedestrianMapAligner


class PreprocessingPipelineRunner:
    """
    Encapsulates the entire preprocessing pipeline, including:
    - Avatar extraction
    - Feature extraction
    - Map alignment
    - Intention binarization
    """

    THRESHOLD_MULTIPLIER = 0.5
    DATA_PATH = PATHS.SAMPLE_DATA_PATH

    def __init__(self):
        # Configure logger
        Logger.configure_unified_logging_file(PATHS.LOGS_PATH / Path("preprocessing.log"))
        self.logger = Logger.get_logger("Preprocessing")
        self.data_path = PreprocessingPipelineRunner.DATA_PATH
        self.threshold_multiplier = PreprocessingPipelineRunner.THRESHOLD_MULTIPLIER

    @staticmethod
    def load_and_binarize_intention(csv_path: str) -> pd.DataFrame:
        """
        Load a CSV file and binarize the 'intended_actions' column.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame with binarized 'intended_actions'.
        """
        df = pd.read_csv(csv_path)
        df["intended_actions"] = df["intended_actions"].apply(
            lambda intention: 1 if intention == "Crossing the road" else 0
        )
        return df

    def run_pedestrian_map_aligner(self) -> None:
        """
        Run the PedestrianMapAligner to construct 3D environments.
        """
        self.logger.info("Running Pedestrian Map Aligner...")
        map_aligner = PedestrianMapAligner(
            scenario_path=None,
            loki_csv_path=PATHS.AVATAR_FILTERED_PEDESTRIANS_CSV_FILE,
            data_path=self.data_path,
        )
        map_aligner.align(
            save=True,
            use_downsampling=True,
            save_path=PATHS.RECONSTRUCTED_DATA_PATH,
            scaling_factor=20,
        )

    def run(self) -> None:
        """
        Execute the entire preprocessing pipeline.
        """
        try:
            # Initialize the avatar extraction pipeline
            pipeline = AvatarExtractionPipeline(
                root_dir=self.data_path,
                csv_path=PATHS.LOKI_CSV_FILE,
                save_dir=PATHS.PEDESTRIAN_AVATARS_PATH,
                threshold_multiplier=self.threshold_multiplier,
            )

            # Load scenario and frame IDs
            scenario_ids, frame_ids = pipeline.load_scenario_frame_ids()
            self.logger.info(f"Loaded {len(scenario_ids)} scenario IDs and {len(frame_ids)} frame IDs.")

            # Validate scenarios and frames
            valid_scenario_ids, valid_frame_ids = pipeline.validate_scenarios_and_frames(scenario_ids, frame_ids)
            self.logger.info(f"Validated scenario IDs: {valid_scenario_ids}")
            self.logger.info(f"Validated frame IDs: {valid_frame_ids}")

            # Process frames and crop pedestrians
            start_time = time.time()
            df_pedestrians_filtered = pipeline.process_all_frames_and_crop_pedestrians(
                valid_scenario_ids, valid_frame_ids
            )
            df_pedestrians_filtered.to_csv(PATHS.AVATAR_FILTERED_PEDESTRIANS_CSV_FILE, index=False)
            self.logger.info(
                f"Time taken to process all frames and crop pedestrians: {time.time() - start_time:.2f} seconds"
            )

            # Binarize intentions and save the updated CSV
            df_binarized = self.load_and_binarize_intention(PATHS.AVATAR_FILTERED_PEDESTRIANS_CSV_FILE)
            df_binarized.to_csv(PATHS.BIN_AVATAR_FILTERED_PEDESTRIANS_CSV_FILE, index=False)

            # Extract pedestrian features
            extractor = PedestrianFeatureExtractor(
                self.data_path,
                PATHS.PEDESTRIAN_FEATURES_PATH,
                PATHS.PEDESTRIAN_AVATARS_PATH
            )
            extractor.extract_features()

            # Run 3D construction
            self.run_pedestrian_map_aligner()

        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user. Exiting gracefully...")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    runner = PreprocessingPipelineRunner()
    runner.run()
