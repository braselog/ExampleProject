import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prepare.py <input_file> <train_output_file> <test_output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    train_output_file = sys.argv[2]
    test_output_file = sys.argv[3]
    
    params = yaml.safe_load(open("params.yaml"))
    seed = params['seed']
    split_ratio = params['prepare']['split_ratio']

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_file), exist_ok=True)

    print(f"Loading raw data from {input_file}")
    df = pd.read_csv(input_file)

    print(f"Splitting data (test ratio: {split_ratio}, seed: {seed})...")
    train_df, test_df = train_test_split(
        df,
        test_size=split_ratio,
        random_state=seed,
        stratify=df['condition'] # Ensure class balance in splits
    )

    print(f"Saving training data to {train_output_file}")
    train_df.to_csv(train_output_file, index=False)
    
    print(f"Saving test data to {test_output_file}")
    test_df.to_csv(test_output_file, index=False)