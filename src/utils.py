import os
import numpy as np

def load_npy_segments(folder_path):
    """
    Load all .npy segment files and ensure numeric 1-D arrays.
    Non-numeric values are removed. Empty segments are skipped.
    """
    segments = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    for fname in files:
        seg = np.load(os.path.join(folder_path, fname), allow_pickle=True)
        seg = np.array(seg).flatten()
        # Keep only numeric values
        seg = np.array([float(x) for x in seg if isinstance(x, (int, float, np.integer, np.floating))])
        if len(seg) > 0:
            segments.append(seg)
    return segments
