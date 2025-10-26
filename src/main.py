from clustering import divide_and_conquer_cluster
from closest_pair import find_closest_pair
from kadane import kadane
from utils import load_dataset, plot_clusters

def main():
    print("=== Time-Series Clustering and Segment Analysis on PulseDB ===")

    # --- Load ABP dataset ---
    segments = load_dataset(
        path="data/VitalDB_AAMI_Test_Subset.mat",  # Path to your .mat file
        signal_type="ABP",
        limit=1000
    )

    # --- Step 1: Divide-and-Conquer Clustering ---
    print("Clustering time-series segments...")
    clusters = divide_and_conquer_cluster(segments, max_size=10, method="dtw")
    print(f" Generated {len(clusters)} clusters.")

    # --- Step 2: Closest Pair per Cluster ---
    for i, cluster in enumerate(clusters):
        pair, dist = find_closest_pair(cluster)
        print(f"Cluster {i+1}: Closest pair distance = {dist:.4f}")

    # --- Step 3: Kadane’s Algorithm ---
    print("\nApplying Kadane’s algorithm on sample signals...")
    for i, signal in enumerate(segments[:5]):
        start, end, max_sum = kadane(signal)
        print(f"Segment {i+1}: Max subarray sum = {max_sum:.4f} (indices {start}-{end})")

    # --- Step 4: Visualization ---
    print("\nSaving representative cluster plots...")
    plot_clusters(clusters)

    print("\n Analysis complete. Plots saved in 'results/cluster_visuals/'.")

if __name__ == "__main__":
    main()
