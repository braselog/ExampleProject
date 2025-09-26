import pandas as pd
import numpy as np # Import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import sys
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns # Using seaborn for easier plotting with hues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a machine learning model')
    parser.add_argument('input_file', help='Path to the training data CSV file')
    parser.add_argument('model_output_file', help='Path to save the trained model')
    parser.add_argument('--feature-importance-plot', help='Path to save the feature importance plot (for applicable models)')
    parser.add_argument('--predicted-probabilities-plot', help='Path to save the predicted probabilities plot')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    model_output_file = args.model_output_file
    feature_importance_plot = args.feature_importance_plot
    predicted_probabilities_plot = args.predicted_probabilities_plot
    
    params = yaml.safe_load(open("params.yaml"))
    seed = params['seed']
    train_params = params['train']
    model_type = train_params['model_type']

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
    if feature_importance_plot:
        os.makedirs(os.path.dirname(feature_importance_plot), exist_ok=True)
    if predicted_probabilities_plot:
        os.makedirs(os.path.dirname(predicted_probabilities_plot), exist_ok=True)

    print(f"Loading training data from {input_file}")
    train_df = pd.read_csv(input_file)
    X_train = train_df.drop('condition', axis=1)
    y_train = train_df['condition']
    feature_names = X_train.columns.tolist()

    print(f"Training model type: {model_type} with seed {seed}...")
    if model_type == 'LogisticRegression':
        model_params = train_params['logreg']
        model = LogisticRegression(random_state=seed, **model_params)
        plot_feature_importance = False
    elif model_type == 'RandomForest':
        model_params = train_params['rf']
        model = RandomForestClassifier(random_state=seed, **model_params)
        plot_feature_importance = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    print(f"Saving model to {model_output_file}")
    joblib.dump(model, model_output_file)

    # --- Plotting Feature Importance (Model Specific - RF) ---
    if feature_importance_plot:
        if plot_feature_importance and hasattr(model, 'feature_importances_'):
            print(f"Generating feature importance plot to {feature_importance_plot}")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances (Random Forest)")
            plt.bar(range(X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(feature_importance_plot)
            plt.close()
        else:
            # Create a placeholder plot for models that don't support feature importance
            print(f"Model does not support feature importance. Creating placeholder plot at {feature_importance_plot}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Feature importance not available\nfor {model_type}', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title(f"Feature Importance - Not Available for {model_type}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(feature_importance_plot)
            plt.close()

    # --- Plotting Predicted Probabilities (Model Agnostic) ---
    if predicted_probabilities_plot:
        if hasattr(model, "predict_proba"):
            print(f"Generating predicted probability distribution plot to {predicted_probabilities_plot}")
            try:
                # Predict probabilities on the training set
                pred_probs_train = model.predict_proba(X_train)

                # Create a DataFrame for easier plotting with seaborn
                # Find the column index corresponding to the 'Diseased' class
                positive_class_label = 'Diseased' # Assuming this is our positive class
                positive_class_index = np.where(model.classes_ == positive_class_label)[0][0]

                prob_df = pd.DataFrame({
                    'Predicted Probability (Class: Diseased)': pred_probs_train[:, positive_class_index],
                    'True Condition': y_train
                })

                plt.figure(figsize=(10, 6))
                # Plot histograms for predicted probabilities, separated by true class
                sns.histplot(data=prob_df, x='Predicted Probability (Class: Diseased)', hue='True Condition', kde=True, bins=30)
                plt.title('Distribution of Predicted Probabilities on Training Data')
                plt.xlabel(f'Predicted Probability of being "{positive_class_label}"')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(predicted_probabilities_plot)
                plt.close()

            except Exception as e:
                print(f"Could not generate probability plot: {e}")
                # Create placeholder plot if there's an error
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f'Error generating probability plot:\n{e}', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.title("Predicted Probabilities - Error")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(predicted_probabilities_plot)
                plt.close()
        else:
            # Create placeholder plot for models that don't support predict_proba
            print(f"Model does not support predict_proba. Creating placeholder plot at {predicted_probabilities_plot}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Predicted probabilities not available\nfor {model_type}', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title(f"Predicted Probabilities - Not Available for {model_type}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(predicted_probabilities_plot)
            plt.close()