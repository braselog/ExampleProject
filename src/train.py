import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier # Import needed if using RF
import joblib
import yaml
from pathlib import Path

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    seed = params['seed']
    train_params = params['train']
    model_type = train_params['model_type']

    input_path = Path("data/processed/train.csv")
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.joblib"

    print(f"Loading training data from {input_path}")
    train_df = pd.read_csv(input_path)
    X_train = train_df.drop('condition', axis=1)
    y_train = train_df['condition']

    print(f"Training model type: {model_type} with seed {seed}...")
    if model_type == 'LogisticRegression':
        model_params = train_params['logreg']
        model = LogisticRegression(random_state=seed, **model_params)
    elif model_type == 'RandomForest':
         model_params = train_params['rf']
         model = RandomForestClassifier(random_state=seed, **model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    print(f"Saving model to {output_path}")
    joblib.dump(model, output_path)