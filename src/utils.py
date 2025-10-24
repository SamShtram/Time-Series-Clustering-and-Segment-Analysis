import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

def load_dataset(path="data/VitalDB_Train_Subset.mat", signal_type="ABP", limit=1000):
    """
    Load ABP/ECG/PPG time-series segments from PulseDB or VitalDB .mat files.
    Each file contains multiple 10-second segments.
    """
    print(f"📂 Loading {signal_type} signals from: {path}")
    data = loadmat(path)

    # Try to find key matching ABP, ECG, or PPG
    key_candidates = [k for k in data.keys() if signal_type.lower() in k.lower()]
    if not key_candidates:
        raise KeyError(f"No '{signal_type}' signal found in {path}. Keys: {list(data.keys())}")

    key = key_candidates[0]
    signals = data[key]

    segments = []
    for i, s in enumerate(signals):
        if len(segments) >= limit:
            break
        try:
            segment = np.array(s).flatten()
            if np.any(np.isnan(segment)):
                continue
            # Normalize (zero mean, unit variance)
            segment = (segment - np.mean(segment)) / np.std(segment)
            segments.append(segment)
        except Exception:
            continue

    print(f"✅ Loaded {len(segments)} {signal_type} segments from {os.path.basename(path)}")
    return segments

def plot_clusters(clusters):
    """
    Plot and save representative signals from each cluster.
    """
    os.makedirs("results/cluster_visuals", exist_ok=True)
    sns.set(style="whitegrid")

    for i, cluster in enumerate(clusters):
        plt.figure(figsize=(10, 4))
        for s in cluster[:3]:  # up to 3 signals
            plt.plot(s, alpha=0.7)
        plt.title(f"Cluster {i+1} (n={len(cluster)})")
        plt.tight_layout()
        plt.savefig(f"results/cluster_visuals/cluster_{i+1}.png")
        plt.close()
