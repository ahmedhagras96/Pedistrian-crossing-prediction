import pandas as pd
import time
from pathlib import Path

from ped_3D_avatar.pedistrian_pc_avatar_extraction import PedestrianProcessingPipeline
from features_extraction.pedistrian_featuers_extraction import extract_pedistrian_featuers
from reconstruction.modules.pedestrian_map_aligner import PedestrianMapAligner
from reconstruction.modules.utils.logger import Logger

# Constants
SCRIPT_PATH = Path(__file__).resolve().parent
LOKI_PATH = SCRIPT_PATH / "LOKI"
LOG_FILE = SCRIPT_PATH / "logs" / "logs.log"
THRESHOLD_MULTIPLIER = 0.5

# Output directories
AVATAR_OUTPUT_DIR = LOKI_PATH / "training_data" / "pedistrian_avatars"
FEATURES_OUTPUT_DIR = LOKI_PATH / "training_data" / "pedistrian_featuers"
CONSTRUCTED_OUTPUT_DIR = LOKI_PATH / "training_data" / "3d_constructed"

# Configure logger
Logger.configure_unified_file_logging(LOG_FILE)
logger = Logger.get_logger(__name__)

def load_and_binarize_intention(csv_path):
    """
    Load a CSV file and binarize the 'intended_actions' column.
    Args:
        csv_path (str or Path): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame with binarized 'intended_actions'.
    """
    df = pd.read_csv(csv_path)
    df["intended_actions"] = df["intended_actions"].apply(
        lambda intention: 1 if intention == "Crossing the road" else 0
    )
    return df

def run_pedestrian_map_aligner():
    """
    Run the PedestrianMapAligner to construct 3D environments.
    """
    logger.info("Running Pedestrian Map Aligner...")
    map_aligner = PedestrianMapAligner(
        scenario_path=None,
        loki_csv_path=LOKI_PATH / "avatar_filtered_pedistrians.csv",
        data_path=LOKI_PATH,
    )
    map_aligner.align(
        save=True,
        use_downsampling=True,
        save_path=CONSTRUCTED_OUTPUT_DIR,
        scaling_factor=20,
    )

def main():
    try:
        # Initialize the pedestrian processing pipeline
        pipeline = PedestrianProcessingPipeline(
            root_dir=LOKI_PATH,
            csv_path=LOKI_PATH / "b_loki.csv",
            save_dir=AVATAR_OUTPUT_DIR,
            threshold_multiplier=THRESHOLD_MULTIPLIER,
        )

        # Load and verify scenario and frame IDs
        scenario_ids, frame_ids = pipeline.load_scenario_frame_ids()
        valid_scenario_ids = pipeline.verify_scenarios(scenario_ids)
        valid_frame_ids = pipeline.verify_frames(valid_scenario_ids, frame_ids)

        # Process frames and crop pedestrians
        start_time = time.time()
        df_pedestrians_filtered = pipeline.process_all_frames_and_crop_pedestrians(
            valid_scenario_ids, valid_frame_ids
        )
        df_pedestrians_filtered.to_csv(LOKI_PATH / "avatar_filtered_pedistrians.csv", index=False)
        logger.info(
            f"Time taken to process all frames and crop pedestrians: {time.time() - start_time:.2f} seconds"
        )

        # Binarize intentions and save the updated CSV
        df_binarized = load_and_binarize_intention(LOKI_PATH / "avatar_filtered_pedistrians.csv")
        df_binarized.to_csv(LOKI_PATH / "b_avatar_filtered_pedistrians.csv", index=False)

        # Extract pedestrian features
        extract_pedistrian_featuers(LOKI_PATH, FEATURES_OUTPUT_DIR, AVATAR_OUTPUT_DIR)

        # Run 3D construction
        run_pedestrian_map_aligner()

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user. Exiting gracefully...")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()