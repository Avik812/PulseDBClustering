from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def divide_and_conquer_clustering(segments, threshold=10):
    """
    Recursively cluster segments using a divide-and-conquer approach based on DTW.
    """
    if len(segments) <= 1:
        return [segments]

    # Compute pairwise DTW distances
    n = len(segments)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            a = np.array(segments[i]).flatten()
            b = np.array(segments[j]).flatten()
            dist, _ = fastdtw(a, b, dist=euclidean)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    median_dist = np.median(dist_matrix)

    # If segments are very dissimilar, split into halves
    if median_dist > threshold and n > 1:
        cluster1 = segments[:n//2]
        cluster2 = segments[n//2:]
        return divide_and_conquer_clustering(cluster1, threshold) + \
               divide_and_conquer_clustering(cluster2, threshold)
    else:
        return [segments]
