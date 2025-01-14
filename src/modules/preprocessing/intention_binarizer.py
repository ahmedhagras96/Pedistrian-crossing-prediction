import pandas as pd

from modules.utilities.file_utils import FileUtils
from modules.utilities.logger import LoggerUtils


class IntentionBinarizer:
    """
    A utility class for loading and preprocessing datasets.
    """

    def __init__(self):
        """Initialize the DataLoader and set up logging."""
        self.logger = LoggerUtils.get_logger(self.__class__.__name__)

    def load_and_binarize_intention(self, file_path):
        """
        Load the CSV dataset and preprocess the 'Intention' column.

        Args:
            file_path (str): The file path to the CSV dataset.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with binary 'Intention' column.
        """
        try:
            self.logger.info(f"Loading dataset from {file_path}")
            df = FileUtils.read_csv(file_path)
            self.logger.info(f"Dataset loaded with {len(df)} rows")

            self.logger.info("Binarizing 'Intention' column")
            df['Intention'] = df['Intention'].apply(
                lambda intention: "Crossing" if intention == "Crossing the road" else "Not Crossing"
            )

            return df
        except Exception as e:
            self.logger.error(f"Error loading or binarizing dataset: {e}")
            raise e
