from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def closest_pair(cluster):
    """
    Find the closest pair of segments in a cluster using DTW.
    """
    min_dist = float('inf')
    pair = (None, None)
    n = len(cluster)

    for i in range(n):
        for j in range(i+1, n):
            seg_i = np.array(cluster[i]).flatten()
            seg_j = np.array(cluster[j]).flatten()
            dist, _ = fastdtw(seg_i, seg_j, dist=euclidean)
            if dist < min_dist:
                min_dist = dist
                pair = (i, j)
    return pair, min_dist
