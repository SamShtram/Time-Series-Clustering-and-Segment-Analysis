import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm

def similarity(ts1, ts2, method="dtw"):
    """Compute similarity (lower = more similar)."""
    if method == "correlation":
        corr = np.corrcoef(ts1, ts2)[0, 1]
        return 1 - corr
    elif method == "euclidean":
        return euclidean(ts1, ts2)
    else:  # DTW
        dist, _ = fastdtw(ts1, ts2)
        return dist

def select_pivot(segments):
    idx = np.random.randint(0, len(segments))
    return segments[idx]

def divide_and_conquer_cluster(segments, depth=0, max_size=10, method="dtw"):
    """Recursively cluster time-series using divide-and-conquer."""
    if len(segments) <= max_size:
        return [segments]

    pivot = select_pivot(segments)
    distances = [similarity(pivot, s, method) for s in tqdm(segments, disable=(depth > 0))]
    median = np.median(distances)

    left = [s for i, s in enumerate(segments) if distances[i] <= median]
    right = [s for i, s in enumerate(segments) if distances[i] > median]

    clusters_left = divide_and_conquer_cluster(left, depth + 1, max_size, method)
    clusters_right = divide_and_conquer_cluster(right, depth + 1, max_size, method)

    return clusters_left + clusters_right
