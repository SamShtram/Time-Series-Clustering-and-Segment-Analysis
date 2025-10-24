import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(path):
    """
    Load a CSV file of PulseDB segments.
    Each row should represent a single time-series segment.
    """
    df = pd.read_csv(path)
    return [df.iloc[i, :].values for i in range(len(df))]

def plot_clusters(clusters):
    """
    Plot representative signals from each cluster.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    for i, cluster in enumerate(clusters):
        plt.subplot(len(clusters), 1, i + 1)
        for s in cluster[:3]:  # plot up to 3 samples per cluster
            plt.plot(s, alpha=0.6)
        plt.title(f"Cluster {i+1} (n={len(cluster)})")

    plt.tight_layout()
    plt.show()

def normalize_signal(signal):
    """Normalize a signal to zero mean and unit variance."""
    signal = np.array(signal)
    return (signal - np.mean(signal)) / np.std(signal)

