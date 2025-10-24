import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from scipy.io import loadmat

def load_dataset(path="data/VitalDB_Train_Subset.mat", signal_type="ABP", limit=1000):
    """
    Load ABP/ECG/PPG time-series segments from PulseDB or VitalDB .mat files.
    Works for both MATLAB v7.3 (HDF5) and older versions.
    Normalizes each signal to zero mean and unit variance.
    """
    print(f"Loading {signal_type} signals from: {path}")

    signals = None

    # Try normal .mat loading first (non-HDF5)
    try:
        data = loadmat(path)
        key_candidates = [k for k in data.keys() if signal_type.lower() in k.lower()]
        if not key_candidates:
            raise KeyError
        key = key_candidates[0]
        signals = data[key]
        print(f"Loaded (non-HDF5) {signal_type} data using SciPy.")
    except (NotImplementedError, KeyError):
        # Handle MATLAB v7.3 (HDF5-based)
        print("Detected MATLAB v7.3 (HDF5) file — using h5py loader.")
        with h5py.File(path, "r") as f:
            key_candidates = [k for k in f.keys() if signal_type.lower() in k.lower()]
            if not key_candidates:
                raise KeyError(f"No '{signal_type}' dataset found in {path}. Keys: {list(f.keys())}")
            key = key_candidates[0]
            signals = np.array(f[key])

    # Process and normalize signals
    segments = []
    for i, s in enumerate(signals):
        if len(segments) >= limit:
            break
        try:
            segment = np.array(s).flatten()
            if np.any(np.isnan(segment)):
                continue
            segment = (segment - np.mean(segment)) / np.std(segment)
            segments.append(segment)
        except Exception:
            continue

    print(f"Loaded {len(segments)} {signal_type} segments from {os.path.basename(path)}")
    return segments


def plot_clusters(clusters):
    """
    Plot and save representative signals from each cluster.
    Automatically saves plots into results/cluster_visuals/.
    """
    os.makedirs("results/cluster_visuals", exist_ok=True)
    sns.set(style="whitegrid")

    for i, cluster in enumerate(clusters):
        plt.figure(figsize=(10, 4))
        for s in cluster[:3]:  # Plot up to 3 sample signals
            plt.plot(s, alpha=0.7)
        plt.title(f"Cluster {i+1} (n={len(cluster)})")
        plt.xlabel("Time (samples)")
        plt.ylabel("Normalized amplitude")
        plt.tight_layout()

        save_path = f"results/cluster_visuals/cluster_{i+1}.png"
        plt.savefig(save_path)
        plt.close()

    print("Saved cluster plots to results/cluster_visuals/")
