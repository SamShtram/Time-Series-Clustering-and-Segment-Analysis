import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from scipy.io import loadmat

def load_dataset(path="data/VitalDB_AAMI_Test_Subset.mat", signal_type="ABP", limit=1000):
    """
    Load ABP/ECG/PPG time-series segments from VitalDB or PulseDB .mat files.
    Supports MATLAB v7.3 (HDF5) and older formats.
    For VitalDB, extracts from 'Subset/Signals' (shape: time x channels x segments).
    """
    print(f"Loading {signal_type} signals from: {path}")

    signals = None

    try:
        # Try normal MAT (non-HDF5)
        data = loadmat(path)
        key_candidates = [k for k in data.keys() if signal_type.lower() in k.lower()]
        if not key_candidates:
            raise KeyError
        key = key_candidates[0]
        signals = data[key]
        print(f"Loaded (non-HDF5) {signal_type} data using SciPy.")
    except (NotImplementedError, KeyError):
        # Handle MATLAB v7.3 (HDF5)
        print("Detected MATLAB v7.3 (HDF5) file — using h5py loader.")
        with h5py.File(path, "r") as f:
            if "Subset" in f.keys():
                subset = f["Subset"]
                if "Signals" not in subset.keys():
                    raise KeyError(f"No 'Signals' dataset found under 'Subset'. Keys: {list(subset.keys())}")
                signals = np.array(subset["Signals"])  # shape = (time, channels, segments)
                print(f"Signals shape: {signals.shape}")
            else:
                raise KeyError(f"No 'Subset' group found in {path}. Keys: {list(f.keys())}")

    # Extract desired signal type from channel index
    if signals.ndim == 3:  # time x channel x segment
        if signal_type.upper() == "ABP":
            idx = 0
        elif signal_type.upper() == "ECG":
            idx = 1
        elif signal_type.upper() == "PPG":
            idx = 2
        else:
            raise ValueError("Invalid signal_type. Choose from 'ABP', 'ECG', 'PPG'.")

        selected = signals[:, idx, :]
        segments_raw = [selected[:, i] for i in range(selected.shape[1])]
    else:
        segments_raw = [np.array(s).flatten() for s in signals]

    # Normalize and clean
    segments = []
    for s in segments_raw[:limit]:
        s = s.astype(float)
        if np.any(np.isnan(s)):
            continue
        s = (s - np.mean(s)) / np.std(s)
        segments.append(s)

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
        for s in cluster[:3]:
            plt.plot(s, alpha=0.7)
        plt.title(f"Cluster {i+1} (n={len(cluster)})")
        plt.xlabel("Time (samples)")
        plt.ylabel("Normalized amplitude")
        plt.tight_layout()

        save_path = f"results/cluster_visuals/cluster_{i+1}.png"
        plt.savefig(save_path)
        plt.close()

    print("Saved cluster plots to results/cluster_visuals/")
