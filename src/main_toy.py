import numpy as np
from clustering import divide_and_conquer_cluster
from closest_pair import find_closest_pair
from kadane import kadane
from utils import plot_clusters

# --- Generate toy signals ---
t = np.linspace(0, 2*np.pi, 100)
toy_segments = [
    np.sin(t),
    np.sin(t + 0.2),
    np.sin(t + 0.4),
    np.sign(np.sin(t)),
    np.sign(np.sin(t + 0.2)),
    np.sign(np.sin(t + 0.4))
]

# --- Run clustering ---
clusters = divide_and_conquer_cluster(toy_segments, max_size=3, method="dtw")
print(f"Generated {len(clusters)} clusters.")

# --- Find closest pairs ---
for i, cluster in enumerate(clusters):
    pair, dist = find_closest_pair(cluster)
    print(f"Cluster {i+1}: Closest pair distance = {dist:.4f}")

# --- Run Kadane’s algorithm on a few signals ---
for i, signal in enumerate(toy_segments[:3]):
    start, end, max_sum = kadane(signal)
    print(f"Segment {i+1}: Max subarray sum = {max_sum:.4f} (indices {start}-{end})")

# --- Plot toy clusters ---
plot_clusters(clusters)