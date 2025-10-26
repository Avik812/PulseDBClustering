import os
from utils import load_npy_segments
from clustering import divide_and_conquer_clustering
from closest_pair import closest_pair
from max_subarray import kadane

DATA_FOLDER = "./data/processed_segments"

# Step 1: Load segments
segments = load_npy_segments(DATA_FOLDER)
print(f"Loaded {len(segments)} numeric segments.")

# Step 2: Cluster using divide-and-conquer
clusters = divide_and_conquer_clustering(segments, threshold=10)
print(f"Total clusters formed: {len(clusters)}")

# Step 3: Closest pair in each cluster
for idx, c in enumerate(clusters):
    if len(c) > 1:
        pair, dist = closest_pair(c)
        print(f"Cluster {idx}: Closest pair indices {pair}, DTW distance {dist}")

# Step 4: Maximum subarray for each segment
for i, seg in enumerate(segments):
    if len(seg) == 0:
        print(f"Segment {i} is empty, skipping.")
        continue
    max_sum, start, end = kadane(seg)
    print(f"Segment {i}: max subarray sum {max_sum}, start {start}, end {end}")
