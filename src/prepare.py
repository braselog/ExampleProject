import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    seed = params['seed']
    split_ratio = params['prepare']['split_ratio']

    input_path = Path("data/raw/measurements.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_train_path = output_dir / "train.csv"
    output_test_path = output_dir / "test.csv"

    print(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)

    print(f"Splitting data (test ratio: {split_ratio}, seed: {seed})...")
    train_df, test_df = train_test_split(
        df,
        test_size=split_ratio,
        random_state=seed,
        stratify=df['condition'] # Ensure class balance in splits
    )

    print(f"Saving processed data to {output_dir}")
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)