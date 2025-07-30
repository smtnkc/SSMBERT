import os
import pandas as pd
from sklearn.utils import shuffle

def shuffle_dataset(dataset, seed=42):
    """
    Shuffle a dataset and save it to a new CSV file.
    
    Args:
        dataset (str): Dataset prefix (e.g., 'case1')
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Shuffled dataframe
    """
    # Setup paths
    input_csv = f"data/{dataset}.csv"
    output_csv = f"data/{dataset}-shuffled-data.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if input file exists
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found.")
    
    # Read and shuffle dataset
    df = pd.read_csv(input_csv)

    # n_rows = len(df)
    # Drop duplicates on 'requirement_en', keep first occurrence
    # df = df.drop_duplicates(subset=['requirement_en'], keep='first').reset_index(drop=True)
    # print(f"Removed duplicates, number of rows before: {n_rows}, after: {len(df)}")

    # Shuffle the dataframe
    df_shuffled = shuffle(df, random_state=seed).reset_index(drop=True)
    
    # Save shuffled dataset
    df_shuffled.to_csv(output_csv, index=False)
    print(f"Shuffled dataset saved to '{output_csv}'")
    
    return df_shuffled

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Shuffle dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        default='case1',
        help='Dataset prefix (without extension)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    shuffle_dataset(args.dataset, args.seed) 