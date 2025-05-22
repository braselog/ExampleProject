import pandas as pd
import numpy as np # Import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
from pathlib import Path
from dvclive import Live
import matplotlib.pyplot as plt
import seaborn as sns # Using seaborn for easier plotting with hues

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    live = Live(save_dvc_exp=True)
    live.log_params(params)
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
    elif model_type == 'RandomForest':
        model_params = train_params['rf']
        model = RandomForestClassifier(random_state=seed, **model_params)
        plot_feature_importance = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)

    print(f"Saving model to {model_output_path}")
    joblib.dump(model, model_output_path)

    # --- Plotting Feature Importance (Model Specific - RF) ---
    if plot_feature_importance and hasattr(model, 'feature_importances_'):
        print(f"Generating feature importance plot to {feature_importance_path}")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances (Random Forest)")
        plt.bar(range(X_train.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(feature_importance_path)
        live.log_image("feature_importance.png", feature_importance_path)
        plt.close()
    elif plot_feature_importance:
         print("Model selected for feature importance plot, but does not support it directly.")

    # --- Plotting Predicted Probabilities (Model Agnostic) ---
    if hasattr(model, "predict_proba"):
        print(f"Generating predicted probability distribution plot to {pred_prob_path}")
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
            plt.savefig(pred_prob_path)
            live.log_image("predicted_probabilities_train.png", pred_prob_path)
            plt.close()

        except Exception as e:
            print(f"Could not generate probability plot: {e}")
    else:
        print("Model does not support 'predict_proba', skipping probability distribution plot.")