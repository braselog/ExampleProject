import pandas as pd
import numpy as np # Import numpy
from sklearn.linear_model import LogisticRegression
import joblib
import yaml
from pathlib import Path


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    seed = params['seed']
    train_params = params['train']
    model_type = train_params['model_type']

    input_path = Path("data/processed/train.csv")
    model_output_dir = Path("models")
    plots_output_dir = Path("plots")

    # Create directories if they don't exist
    model_output_dir.mkdir(parents=True, exist_ok=True)
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    model_output_path = model_output_dir / "model.joblib"
    feature_importance_path = plots_output_dir / "feature_importance.png"
    # --- Define path for the new model-agnostic plot ---
    pred_prob_path = plots_output_dir / "predicted_probabilities_train.png"

    print(f"Loading training data from {input_path}")
    train_df = pd.read_csv(input_path)
    X_train = train_df.drop('condition', axis=1)
    y_train = train_df['condition']
    feature_names = X_train.columns.tolist()

    print(f"Training model type: {model_type} with seed {seed}...")
    if model_type == 'LogisticRegression':
        model_params = train_params['logreg']
        model = LogisticRegression(random_state=seed, **model_params)
        plot_feature_importance = False

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    print(f"Saving model to {model_output_path}")
    joblib.dump(model, model_output_path)

