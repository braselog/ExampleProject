import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path

def simulate_data(n_samples_per_class, n_features, means, stds, seed):
    """Generates simulated measurement data for two classes."""
    np.random.seed(seed)
    data = []
    labels = []

    # Healthy samples
    healthy_data = np.random.normal(loc=means['healthy'], scale=stds['healthy'], size=(n_samples_per_class, n_features))
    data.append(healthy_data)
    labels.extend(['Healthy'] * n_samples_per_class)

    # Diseased samples
    diseased_data = np.random.normal(loc=means['diseased'], scale=stds['diseased'], size=(n_samples_per_class, n_features))
    data.append(diseased_data)
    labels.extend(['Diseased'] * n_samples_per_class)

    X = np.vstack(data)
    y = np.array(labels)

    feature_names = [f'gene_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['condition'] = y

    return df

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))['simulate']
    seed = yaml.safe_load(open("params.yaml"))['seed']

    os.makedirs(os.path.dirname(params['out_file']), exist_ok=True)
    output_path = params['out_file']

    means = {'healthy': params['mean_healthy'], 'diseased': params['mean_diseased']}
    stds = {'healthy': params['std_healthy'], 'diseased': params['std_diseased']}

    print(f"Simulating data with seed {seed}...")
    df = simulate_data(
        n_samples_per_class=params['n_samples_per_class'],
        n_features=params['n_features'],
        means=means,
        stds=stds,
        seed=seed
    )

    print(f"Saving raw data to {output_path}")
    df.to_csv(output_path, index=False)