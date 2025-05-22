# Machine Learning Pipeline for Synthetic Classification Data

This project implements a DVC-managed machine learning pipeline that:
1.  Simulates synthetic data for a binary classification problem (e.g., "Healthy" vs. "Diseased").
2.  Prepares and preprocesses the data.
3.  Trains a classification model.
4.  Evaluates the model's performance and generates relevant plots and metrics.

The pipeline is configurable through `params.yaml` and uses `dvc` to manage stages and outputs.

## Project Structure

Key directories and files in the project include:
- `data/`: Contains raw, intermediate, and processed data.
  - `data/raw/`: Stores the initially simulated data (`measurements.csv`).
  - `data/processed/`: Stores the split training (`train.csv`) and testing (`test.csv`) datasets.
- `models/`: Stores the trained machine learning model (`model.joblib`).
- `src/`: Contains the Python scripts for each pipeline stage.
  - `src/simulate.py`: Generates the raw data.
  - `src/prepare.py`: Splits data into train and test sets.
  - `src/train.py`: Trains the model.
  - `src/evaluate.py`: Evaluates the model and generates metrics.
- `plots/`: Contains generated plots like feature importance, confusion matrix, ROC curve, and predicted probability distributions.
- `metrics/`: Stores evaluation metrics in `metrics.json`.
- `dvc.yaml`: Defines the DVC pipeline stages, their inputs, outputs, and dependencies.
- `params.yaml`: Contains parameters for data simulation, data preparation, and model training.
- `requirements.txt`: Lists the Python dependencies for this project.
- `.dvc/`: Directory used by DVC to track data, models, and metrics. (Usually not modified manually).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project uses `pip` for dependency management. Install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```
    This will install packages such as pandas, scikit-learn, PyYAML, joblib, and numpy.

4.  **Initialize DVC (if you plan to modify or track new data/models):**
    This project uses DVC to manage data, models, and pipeline stages. If you haven't already, you might need to initialize DVC in your environment.
    ```bash
    dvc init
    ```
    *Note: If you only intend to run the existing pipeline, `dvc init` might not be strictly necessary if the `.dvc` directory is already present and configured from the clone.*

## Pipeline Stages and Execution

This project uses DVC to define and manage the machine learning pipeline. The stages are defined in `dvc.yaml`.

**Key Pipeline Stages:**

*   **`simulate`**: Generates synthetic raw data.
    *   *Script*: `src/simulate.py`
    *   *Output*: `data/raw/measurements.csv`
*   **`prepare`**: Splits the raw data into training and testing sets.
    *   *Script*: `src/prepare.py`
    *   *Inputs*: `data/raw/measurements.csv`
    *   *Outputs*: `data/processed/train.csv`, `data/processed/test.csv`
*   **`train`**: Trains the classification model using the training data.
    *   *Script*: `src/train.py`
    *   *Inputs*: `data/processed/train.csv`
    *   *Outputs*: `models/model.joblib`, plots in `plots/` (e.g., `feature_importance.png`, `predicted_probabilities_train.png`)
*   **`evaluate`**: Evaluates the trained model using the test data and generates metrics.
    *   *Script*: `src/evaluate.py`
    *   *Inputs*: `models/model.joblib`, `data/processed/test.csv`
    *   *Outputs*: `metrics/metrics.json`, plots in `plots/` (e.g., `confusion_matrix.png`, `roc_curve.png`)

**Running the Pipeline:**

*   **To run the entire pipeline:**
    This command will reproduce all stages defined in `dvc.yaml` if their inputs or parameters have changed, or if their outputs are missing.
    ```bash
    dvc repro
    ```

*   **To run a specific stage and its dependencies:**
    For example, to run only the `train` stage (and any necessary preceding stages):
    ```bash
    dvc repro train
    ```

*   **To force re-running a stage (even if inputs/params haven't changed):**
    ```bash
    dvc repro -f <stage_name>
    ```
    For example:
    ```bash
    dvc repro -f train
    ```

## Configuration

The project's behavior and parameters for various stages can be controlled through the `params.yaml` file. This file allows you to modify aspects of data simulation, data preparation, and model training without altering the source code.

**Key configurable parameters in `params.yaml` include:**

*   **`seed`**: A global random seed for ensuring reproducibility across all stages.
*   **`simulate`**: Parameters for data generation (`src/simulate.py`):
    *   `n_samples_per_class`: Number of samples to generate for each class.
    *   `n_features`: Number of features (e.g., genes) in the simulated data.
    *   `mean_healthy`, `std_healthy`: Mean and standard deviation for the "Healthy" class data distribution.
    *   `mean_diseased`, `std_diseased`: Mean and standard deviation for the "Diseased" class data distribution.
*   **`prepare`**: Parameters for data splitting (`src/prepare.py`):
    *   `split_ratio`: The proportion of the dataset to be used as the test set.
*   **`train`**: Parameters for model training (`src/train.py`):
    *   `model_type`: Specifies the type of model to train (e.g., `'LogisticRegression'`, `'RandomForest'`).
    *   `logreg`: Contains parameters specific to Logistic Regression (e.g., `C` for regularization).
    *   `rf`: Contains parameters specific to Random Forest (e.g., `n_estimators`, `max_depth`).

**Example: Changing the model type**

To switch from Logistic Regression to Random Forest, you would modify the `train.model_type` parameter in `params.yaml`:

```yaml
train:
  model_type: 'RandomForest' # Changed from 'LogisticRegression'
  # ... other parameters ...
```

After changing parameters in `params.yaml`, you can re-run the relevant pipeline stages using `dvc repro` for the changes to take effect. For instance, if you change training parameters, running `dvc repro train` (or just `dvc repro`) will retrain the model with the new settings.

## Outputs

After running the pipeline, the following outputs will be generated and tracked by DVC:

*   **Processed Data:**
    *   `data/processed/train.csv`: The dataset used for training the model.
    *   `data/processed/test.csv`: The dataset used for evaluating the model.

*   **Trained Model:**
    *   `models/model.joblib`: The serialized, trained machine learning model. This can be a Logistic Regression or Random Forest model, depending on the configuration in `params.yaml`.

*   **Evaluation Metrics:**
    *   `metrics/metrics.json`: A JSON file containing key performance metrics of the model on the test set. This includes:
        *   `accuracy`
        *   `precision_diseased`
        *   `recall_diseased`
        *   `f1_score_diseased`
        *   `roc_auc` (if applicable for the model)

*   **Plots:**
    Located in the `plots/` directory, these visualizations help in understanding the data and model performance:
    *   `plots/feature_importance.png`: Shows the relative importance of each feature if a RandomForest model is trained and supports this. (Generated by `src/train.py`)
    *   `plots/predicted_probabilities_train.png`: A histogram showing the distribution of predicted probabilities on the training data, separated by true class. Helps to see how well the model separates classes on data it has seen. (Generated by `src/train.py`)
    *   `plots/confusion_matrix.png`: Visualizes the performance of the classification model, showing true positives, true negatives, false positives, and false negatives. (Generated by `src/evaluate.py`)
    *   `plots/roc_curve.png`: Displays the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) value, illustrating the diagnostic ability of the classifier. (Generated by `src/evaluate.py`, if applicable)

These outputs can be accessed directly from the respective paths after the pipeline has been successfully executed.

## Contributing and Future Improvements

This project serves as a template and demonstration of a DVC-based machine learning pipeline. Contributions and suggestions for improvements are welcome!

**Potential Future Improvements:**

*   **Experiment with more models:** Integrate and test other classification algorithms (e.g., SVM, Gradient Boosting).
*   **Hyperparameter optimization:** Add a DVC stage for hyperparameter tuning (e.g., using GridSearchCV or Optuna).
*   **More sophisticated data simulation:** Allow for more complex relationships between features and classes in `src/simulate.py`.
*   **Advanced evaluation:** Include more detailed evaluation plots or metrics.
*   **CI/CD integration:** Set up a CI/CD pipeline (e.g., using GitHub Actions) to automatically run tests and the DVC pipeline on changes.
*   **Data validation:** Add a stage for validating the input data schema and characteristics.

**How to Contribute:**

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix (e.g., `git checkout -b feature/my-new-feature` or `git checkout -b fix/issue-tracker-bug`).
3.  **Make your changes.**
4.  **Ensure your changes pass any existing tests, or add new tests if applicable.**
5.  **Commit your changes** with a clear and descriptive commit message.
6.  **Push your branch** to your forked repository.
7.  **Open a Pull Request** to the main repository, detailing the changes you've made.

Feel free to open an issue to discuss potential changes or to report bugs.
