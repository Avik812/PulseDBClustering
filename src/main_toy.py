import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# -----------------------------
# Max Subarray (Kadane's Algorithm)
# -----------------------------
def kadane(arr):
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
# Closest Pair using DTW
# -----------------------------
def closest_pair(cluster):
    min_dist = float('inf')
    pair = (None, None)
    n = len(cluster)
    for i in range(n):
        for j in range(i+1, n):
            a = np.ravel(cluster[i])  # Ensure 1-D
            b = np.ravel(cluster[j])
            dist, _ = fastdtw(a, b, dist=euclidean)
            if dist < min_dist:
                min_dist = dist
                pair = (i, j)
    return pair, min_dist

# -----------------------------
# Divide-and-Conquer Clustering
# -----------------------------
def divide_and_conquer_clustering(segments, threshold=5):
    if len(segments) <= 1:
        return [segments]

    n = len(segments)
    dist_matrix = np.zeros((n, n))
    # compute pairwise DTW distances
    for i in range(n):
        for j in range(i+1, n):
            a = np.ravel(segments[i])
            b = np.ravel(segments[j])
            dist, _ = fastdtw(a, b, dist=euclidean)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    median_dist = np.median(dist_matrix)
    if median_dist > threshold:
        mid = n // 2
        left = divide_and_conquer_clustering(segments[:mid], threshold)
        right = divide_and_conquer_clustering(segments[mid:], threshold)
        return left + right
    else:
        return [segments]

# -----------------------------
# Toy Data
# -----------------------------
toy_segments = [
    np.array([1,2,3,2,1]),
    np.array([2,3,4,3,2]),
    np.array([10,11,10,11,10]),
    np.array([11,12,11,12,11]),
    np.array([0,1,0,1,0])
]

# Convert all to proper 1-D NumPy arrays
segments = [np.ravel(s) for s in toy_segments]

# Step 1: Cluster
clusters = divide_and_conquer_clustering(segments, threshold=5)
print(f"Total clusters: {len(clusters)}")

# Step 2: Closest pair in each cluster
for idx, cluster in enumerate(clusters):
    if len(cluster) > 1:
        pair, dist = closest_pair(cluster)
        print(f"Cluster {idx}: Closest pair indices {pair}, DTW distance {dist}")

# Step 3: Kadane's max subarray for each segment
for i, seg in enumerate(segments):
    max_sum, start, end = kadane(seg)
    print(f"Segment {i}: max subarray sum {max_sum}, start {start}, end {end}")
