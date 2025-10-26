# -----------------------------
# toy_example_fixed.py
# -----------------------------
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# -----------------------------
# 1. Divide-and-Conquer Clustering (FIXED)
# -----------------------------
def divide_and_conquer_clustering(segments, threshold=5):
    """
    Recursively clusters segments. The function must ONLY work with a flat list
    of segments during its execution, and return a list of clusters (list of lists of segments).
    """
    if len(segments) <= 1:
        # Base case: A cluster is formed. If len is 1, return [[segment]]. If 0, return [].
        return [segments] if segments else []

    n = len(segments)
    dist_matrix = np.zeros((n, n))
    
    # 1. Compute Distance Matrix (This part requires 'a' and 'b' to be 1-D arrays)
    for i in range(n):
        for j in range(i+1, n):
            # Each segment is expected to be a 1-D numpy array of floats
            a = segments[i]
            b = segments[j]
            # If the ValueError occurs here, the input 'a' or 'b' is not 1-D.
            dist, _ = fastdtw(a, b, dist=euclidean)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    median_dist = np.median(dist_matrix[np.triu_indices(n, k=1)]) # Use only upper triangle for median

    # 2. Division/Recursion
    if median_dist > threshold and n > 1:
        # Split the current list of segments into two new lists of segments
        cluster1_segments = segments[:n//2]
        cluster2_segments = segments[n//2:]
        
        # Recursively call and concatenate the resulting LISTS OF CLUSTERS
        return divide_and_conquer_clustering(cluster1_segments, threshold) + \
               divide_and_conquer_clustering(cluster2_segments, threshold)
    else:
        # Base Case: It's a final cluster, return it WRAPPED IN A LIST 
        # to maintain the "list of clusters" structure for concatenation.
        return [segments]

# -----------------------------
# 2. Closest Pair in Cluster (No Change)
# -----------------------------
def closest_pair(cluster):
    min_dist = float('inf')
    pair = (None, None)
    n = len(cluster)
    for i in range(n):
        for j in range(i+1, n):
            a = cluster[i]
            b = cluster[j]
            dist, _ = fastdtw(a, b, dist=euclidean)
            if dist < min_dist:
                min_dist = dist
                pair = (i, j)
    return pair, min_dist

# -----------------------------
# 3. Kadane's Maximum Subarray (No Change)
# -----------------------------
def kadane(arr):
    if len(arr) == 0:
        return 0, None, None

    max_sum = current_sum = arr[0]
    start = end = s = 0
    for i in range(1, len(arr)):
        if current_sum < 0:
            current_sum = arr[i]
            s = i
        else:
            current_sum += arr[i]

        if current_sum > max_sum:
            max_sum = current_sum
            start = s
            end = i
    return max_sum, start, end

# -----------------------------
# Toy segments (ALL 1-D numpy arrays!)
# -----------------------------
toy_segments = [
    np.array([1, 2, 3, 2, 1], dtype=float),
    np.array([2, 3, 4, 3, 2], dtype=float),
    np.array([10, 11, 10, 11, 10], dtype=float),
    np.array([11, 12, 11, 12, 11], dtype=float)
]

# -----------------------------
# Run clustering
# -----------------------------
clusters = divide_and_conquer_clustering(toy_segments, threshold=5)
print("Clusters formed:")
for idx, c in enumerate(clusters):
    print(f"Cluster {idx}: {c}")

# -----------------------------
# Run closest pair
# -----------------------------
print("\nClosest pair in each cluster:")
for idx, c in enumerate(clusters):
    if len(c) > 1:
        pair, dist = closest_pair(c)
        # Note: pair indices are relative to the start of the cluster list 'c'
        print(f"Cluster {idx} -> Pair indices: {pair}, DTW distance: {dist:.2f}")

# -----------------------------
# Run Kadane
# -----------------------------
print("\nMaximum subarray in each segment:")
for idx, seg in enumerate(toy_segments):
    max_sum, start, end = kadane(seg)
    print(f"Segment {idx} -> Max sum: {max_sum}, Start: {start}, End: {end}")