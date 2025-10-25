import numpy as np
from src.utils import load_segments_csv
from src.clustering import divide_and_conquer
from src.closest_pair import closest_pair_dtw
from src.max_subarray import kadane
import matplotlib.pyplot as plt

def run_pipeline(path_csv, out_prefix='output'):
    df = load_segments_csv(path_csv)
    segments = list(df['values'].values)
    ids = list(df['segment_id'].values)

    clusters = divide_and_conquer(segments, indices=list(range(len(segments))), min_size=20)
    print(f"Found {len(clusters)} clusters")

    # For each cluster, compute closest pair and run Kadane on each segment
    report = []
    for ci, cluster in enumerate(clusters):
        segs = [segments[i] for i in cluster]
        if len(segs) >= 2:
            i,j,d = closest_pair_dtw(segs)
            rep_pair = (cluster[i], cluster[j], d)
        else:
            rep_pair = (cluster[0], None, None)

        kadane_results = [kadane(s) for s in segs]
        report.append({
            'cluster_id': ci,
            'members': [ids[i] for i in cluster],
            'rep_pair': rep_pair,
            'kadane': kadane_results
        })

    # quick visualize first cluster
    if clusters:
        first = clusters[0]
        plt.figure(figsize=(8,5))
        for idx in first[:5]:
            plt.plot(segments[idx], alpha=0.6)
        plt.title(f"Cluster 0 (showing up to 5 members)")
        plt.show()

    return report

if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv)>1 else 'data/toy_segments.csv'
    run_pipeline(csv)
