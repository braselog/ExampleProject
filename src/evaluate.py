import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import yaml
from pathlib import Path

if __name__ == "__main__":
    model_path = Path("models/model.joblib")
    test_data_path = Path("data/processed/test.csv")
    output_dir = Path("metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    print(f"Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop('condition', axis=1)
    y_test = test_df['condition']

    print("Making predictions...")
    y_pred = model.predict(X_test)

    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate precision, recall, f1 for the 'Diseased' class (or average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', pos_label='Diseased'
    )

    metrics = {
        'accuracy': accuracy,
        'precision_diseased': precision,
        'recall_diseased': recall,
        'f1_score_diseased': f1
    }

    print(f"Metrics: {metrics}")
    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)