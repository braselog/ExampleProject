# src/evaluate.py
import pandas as pd
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc, # Use auc directly
    ConfusionMatrixDisplay # For plotting confusion matrix
)
import matplotlib.pyplot as plt
import yaml
import sys
import os
from pathlib import Path
import numpy as np # Needed for roc curve processing

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python evaluate.py <test_data_file> <model_file> <metrics_output_file> <confusion_matrix_plot> <roc_curve_plot>")
        sys.exit(1)
    
    test_data_file = sys.argv[1]
    model_file = sys.argv[2]
    metrics_output_file = sys.argv[3]
    confusion_matrix_plot = sys.argv[4]
    roc_curve_plot = sys.argv[5]

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(metrics_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(confusion_matrix_plot), exist_ok=True)
    os.makedirs(os.path.dirname(roc_curve_plot), exist_ok=True)

    print(f"Loading model from {model_file}")
    model = joblib.load(model_file)

    print(f"Loading test data from {test_data_file}")
    test_df = pd.read_csv(test_data_file)
    X_test = test_df.drop('condition', axis=1)
    y_test = test_df['condition']

    # Get class labels from the model or data
    classes = model.classes_ if hasattr(model, 'classes_') else np.unique(y_test)

    print("Making predictions...")
    y_pred = model.predict(X_test)
    # Get probability scores for the positive class (needed for ROC)
    # Assumes the positive class is the second one ('Diseased' in our case)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        print("Model does not support predict_proba, ROC curve cannot be generated.")
        y_pred_proba = None # Set to None if not available

    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
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
    print(f"Saving metrics to {metrics_output_file}")
    with open(metrics_output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # --- Plotting Confusion Matrix ---
    print(f"Generating confusion matrix plot to {confusion_matrix_plot}")
    try:
        # Ensure y_test and y_pred are consistent types (e.g., strings)
        # This can be important if one is inferred as object and other as string
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_test.astype(str),
            y_pred.astype(str),
            display_labels=classes.astype(str), # Ensure labels match type
            cmap=plt.cm.Blues
        )
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(confusion_matrix_plot)
        plt.close()
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")


    # --- Plotting ROC Curve ---
    if y_pred_proba is not None:
        print(f"Generating ROC curve plot to {roc_curve_plot}")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='Diseased')
        roc_auc = auc(fpr, tpr)
        metrics['roc_auc'] = roc_auc # Add AUC to metrics

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_curve_plot)
        plt.close()

        # Re-save metrics JSON now that it includes AUC
        print(f"Re-saving metrics (with AUC) to {metrics_output_file}")
        with open(metrics_output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    else:
        print("Skipping ROC curve generation as predict_proba was not available.")