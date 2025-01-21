import pandas as pd

from modules.config.logger import Logger


class BinaryIntentionSplitter:
    """
    A utility class for loading, preprocessing, and splitting 'Intention' data.
    """

    # Use a class-level logger (class name specified as string for static usage)
    logger = Logger.get_logger("BinaryIntentionSplitter")

    @staticmethod
    def load_and_binarize_intention(csv_path: str) -> pd.DataFrame:
        """
        Load a CSV dataset and binarize the 'Intention' column into two categories:
        "Crossing" and "Not Crossing".

        Args:
            csv_path (str): The file path to the CSV dataset.

        Returns:
            pd.DataFrame: A DataFrame with the binarized 'Intention' column.
        """
        BinaryIntentionSplitter.logger.info(f"Loading and binarizing CSV dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        df['Intention'] = df['Intention'].apply(
            lambda x: "Crossing" if x == "Crossing the road" else "Not Crossing"
        )
        return df

    @staticmethod
    def split_dataframe_into_chunks(df: pd.DataFrame, num_chunks: int) -> list:
        """
        Split a DataFrame into a specified number of equal-sized chunks.

        Args:
            df (pd.DataFrame): The DataFrame to split.
            num_chunks (int): Number of chunks to create.

        Returns:
            list: A list containing the split DataFrames.
        """
        BinaryIntentionSplitter.logger.info(
            f"Splitting DataFrame of length {len(df)} into {num_chunks} chunks"
        )
        rows_per_chunk = len(df) // num_chunks
        return [df.iloc[i * rows_per_chunk: (i + 1) * rows_per_chunk] for i in range(num_chunks)]

    @staticmethod
    def get_corresponding_splits(
            crossing_df: pd.DataFrame, non_crossing_df: pd.DataFrame, num_crossing_chunks: int
    ) -> tuple:
        """
        Split both 'Crossing' and 'Not Crossing' DataFrames into corresponding chunks,
        ensuring each chunk in 'Not Crossing' matches the size of each chunk in 'Crossing'.

        Args:
            crossing_df (pd.DataFrame): DataFrame containing 'Crossing' intentions.
            non_crossing_df (pd.DataFrame): DataFrame containing 'Not Crossing' intentions.
            num_crossing_chunks (int): Number of chunks for the 'Crossing' DataFrame.

        Returns:
            tuple: A tuple of two lists:
                - The split DataFrames for 'Crossing'
                - The split DataFrames for 'Not Crossing'
        """
        BinaryIntentionSplitter.logger.info(
            "Splitting 'Crossing' and 'Not Crossing' data into corresponding chunks"
        )
        crossing_chunks = BinaryIntentionSplitter.split_dataframe_into_chunks(
            crossing_df, num_crossing_chunks
        )
        rows_per_crossing_chunk = len(crossing_chunks[0])
        num_non_crossing_chunks = len(non_crossing_df) // rows_per_crossing_chunk

        non_crossing_chunks = BinaryIntentionSplitter.split_dataframe_into_chunks(
            non_crossing_df, num_non_crossing_chunks
        )

        return crossing_chunks, non_crossing_chunks

    @staticmethod
    def generate_all_matching_split_combinations(
            crossing_df: pd.DataFrame, non_crossing_df: pd.DataFrame, num_crossing_chunks: int
    ) -> dict:
        """
        Generate all possible combinations of splits between 'Crossing' and 'Not Crossing' DataFrames.

        Args:
            crossing_df (pd.DataFrame): DataFrame containing 'Crossing' intentions.
            non_crossing_df (pd.DataFrame): DataFrame containing 'Not Crossing' intentions.
            num_crossing_chunks (int): Number of chunks for the 'Crossing' DataFrame.

        Returns:
            dict: A dictionary with keys as (crossing_index, non_crossing_index) tuples
                  and values as corresponding DataFrame pairs.
        """
        BinaryIntentionSplitter.logger.info(
            "Generating all matching split combinations between 'Crossing' and 'Not Crossing' data"
        )
        split_combinations = {}
        crossing_chunks, non_crossing_chunks = BinaryIntentionSplitter.get_corresponding_splits(
            crossing_df, non_crossing_df, num_crossing_chunks
        )

        for i, crossing_chunk in enumerate(crossing_chunks):
            for j, non_crossing_chunk in enumerate(non_crossing_chunks):
                split_combinations[(i, j)] = (crossing_chunk, non_crossing_chunk)

        return split_combinations

    @staticmethod
    def split_dataframes(df: pd.DataFrame) -> None:
        """
        Separate the data based on 'Intention' and generate combinations of
        equally sized chunks for 'Crossing' vs 'Not Crossing'.

        Args:
            df (pd.DataFrame): The DataFrame containing the 'Intention' column.

        Returns:
            None
        """
        BinaryIntentionSplitter.logger.info("Splitting DataFrame into 'Crossing' and 'Not Crossing' chunks")
        crossing_data = df[df['Intention'] == "Crossing"]
        non_crossing_data = df[df['Intention'] == "Not Crossing"]

        split_combinations = BinaryIntentionSplitter.generate_all_matching_split_combinations(
            crossing_data, non_crossing_data, num_crossing_chunks=3
        )

        for (crossing_idx, non_crossing_idx), (crossing_split, non_crossing_split) in split_combinations.items():
            BinaryIntentionSplitter.logger.info(
                f"Combination (Crossing Split {crossing_idx}, Not Crossing Split {non_crossing_idx}):\n"
                f"Crossing chunk size: {len(crossing_split)}, "
                f"Not Crossing chunk size: {len(non_crossing_split)}"
            )
