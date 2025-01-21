from pathlib import Path

from modules.config.logger import Logger
from modules.config.paths_loader import PATHS
from modules.features.preprocessing.binary_intention_splitter import BinaryIntentionSplitter

#: Module-level logger for this script
logger = Logger.get_logger("FeaturesPreprocessingPipeline")


def run_binary_intention_preprocessing_pipeline() -> None:
    """
    Execute a pipeline that:
      1. Loads data from a CSV file.
      2. Binarizes the 'Intention' column.
      3. Saves the new CSV to a specified path.

    Returns:
        None
    """
    # Define the path to the raw CSV file
    loki_csv_path = PATHS.LOKI_CSV_FILE

    # 1. Load and preprocess to binarize the 'Intention' column
    data_frame = BinaryIntentionSplitter.load_and_binarize_intention(loki_csv_path)

    # 2. Save new CSV as b_loki.csv
    bin_loki_csv_path = PATHS.BIN_LOKI_CSV_FILE
    data_frame.to_csv(bin_loki_csv_path, index=False)
    logger.info(f"Binarized data saved to: {bin_loki_csv_path}")

    # 3. (Optionally) demonstrate splitting into chunks
    # e.g. IntentionDataUtils.split_dataframes(data_frame)


def run_preprocessing_pipeline() -> None:
    """
    Run the complete preprocessing pipeline, which currently
    includes the binary intention preprocessing steps.
    

    Returns:
        None
    """
    run_binary_intention_preprocessing_pipeline()


if __name__ == '__main__':
    """
    Main execution sequence:
      1. Configure unified logging to write to a file.
      2. Execute the preprocessing pipeline.
    """
    # Configure unified logging to write to a file
    Logger.configure_unified_logging_file(
        PATHS.LOGS_PATH / Path("features_preprocessing_pipeline.log")
    )

    # Run the full preprocessing pipeline
    run_preprocessing_pipeline()
