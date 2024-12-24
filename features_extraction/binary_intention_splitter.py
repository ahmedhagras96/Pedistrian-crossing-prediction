import os
import pandas as pd

# Define the path to the dataset
script_path = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
loki_path = os.path.normpath(os.path.abspath(os.path.join(script_path, "..", "LOKI")))
loki_csv_path = os.path.normpath(os.path.abspath(os.path.join(loki_path, "loki.csv")))

def load_and_binarize_intention(path):
    """
    Load the CSV dataset and preprocess the 'Intention' column.

    Parameters:
    path (str): The file path to the CSV dataset.

    Returns:
    DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(path)
    df['Intention'] = df['Intention'].apply(
        lambda intention: "Crossing" if intention == "Crossing the road" else "Not Crossing"
    )
    return df

def split_dataframe_into_chunks(df, num_chunks):
    """
    Split a DataFrame into a specified number of equal-sized chunks.

    Parameters:
    df (DataFrame): The DataFrame to split.
    num_chunks (int): Number of chunks to create.

    Returns:
    list of DataFrames: List containing the split DataFrames.
    """
    rows_per_chunk = len(df) // num_chunks
    return [df.iloc[i * rows_per_chunk : (i + 1) * rows_per_chunk] for i in range(num_chunks)]

def get_corresponding_splits(crossing_df, non_crossing_df, num_crossing_chunks):
    """
    Split both 'Crossing' and 'Not Crossing' DataFrames into corresponding chunks.

    Parameters:
    crossing_df (DataFrame): DataFrame containing 'Crossing' intentions.
    non_crossing_df (DataFrame): DataFrame containing 'Not Crossing' intentions.
    num_crossing_chunks (int): Number of chunks for the 'Crossing' DataFrame.

    Returns:
    tuple: Two lists containing the split DataFrames for 'Crossing' and 'Not Crossing'.
    """
    # Split the 'Crossing' DataFrame
    crossing_chunks = split_dataframe_into_chunks(crossing_df, num_crossing_chunks)
    
    # Determine the number of chunks for 'Not Crossing' based on chunk size
    rows_per_crossing_chunk = len(crossing_chunks[0])
    num_non_crossing_chunks = len(non_crossing_df) // rows_per_crossing_chunk
    non_crossing_chunks = split_dataframe_into_chunks(non_crossing_df, num_non_crossing_chunks)
    
    return crossing_chunks, non_crossing_chunks

def generate_all_matching_split_combinations(crossing_df, non_crossing_df, num_crossing_chunks):
    """
    Generate all possible combinations of splits between 'Crossing' and 'Not Crossing' DataFrames.

    Parameters:
    crossing_df (DataFrame): DataFrame containing 'Crossing' intentions.
    non_crossing_df (DataFrame): DataFrame containing 'Not Crossing' intentions.
    num_crossing_chunks (int): Number of chunks for the 'Crossing' DataFrame.

    Returns:
    dict: Dictionary with keys as (crossing_index, non_crossing_index) tuples and values as corresponding DataFrame pairs.
    """
    split_combinations = {}
 
    crossing_chunks, non_crossing_chunks = get_corresponding_splits(
        crossing_df, non_crossing_df, num_crossing_chunks
    )

    for i, crossing_chunk in enumerate(crossing_chunks):
        for j, non_crossing_chunk in enumerate(non_crossing_chunks):
            split_combinations[(i, j)] = (crossing_chunk, non_crossing_chunk)
    
    return split_combinations
  
def split_dataframes(data_frame):
      # Separate the data based on 'Intention'
    crossing_data = data_frame[data_frame['Intention'] == "Crossing"]
    non_crossing_data = data_frame[data_frame['Intention'] == "Not Crossing"]
    
    # Generate split combinations
    split_combinations = generate_all_matching_split_combinations(crossing_data, non_crossing_data, 3)
    
    # Display information about each split
    for (crossing_idx, non_crossing_idx), (crossing_split, non_crossing_split) in split_combinations.items():
        print(f"Combination (Crossing Split {crossing_idx}, Not Crossing Split {non_crossing_idx}):")
        crossing_split.info()
        non_crossing_split.info()
        print("-" * 50)
    
    # Optionally, save the split DataFrames to CSV files
    # for (i, j), (crossing_split, non_crossing_split) in split_combinations.items():
    #     crossing_split.to_csv(f'crossing_split_{i}.csv', index=False)
    #     non_crossing_split.to_csv(f'not_crossing_split_{j}.csv', index=False)

def main():
    # Load and preprocess the intention to binarize it
    data_frame = load_and_binarize_intention(loki_csv_path)
    
    # Splits into ratio equal batches of crossing and not corssing to counter inbalanced data
    # split_dataframes(data_frame)
    
    # Save new csv as b_loki.csv
    data_frame.to_csv(os.path.join(loki_path, "b_loki.csv"), index=False)
    

if __name__ == "__main__":
    main()
