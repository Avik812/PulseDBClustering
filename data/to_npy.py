import os
import h5py
import numpy as np

DATA_PATH = "/Users/avik/Documents/projects/PulseDBClustering/data/segments"
OUT_PATH = "/Users/avik/Documents/projects/PulseDBClustering/data/processed_segments"
os.makedirs(OUT_PATH, exist_ok=True)

for fname in os.listdir(DATA_PATH):
    if not fname.endswith(".mat"):
        continue
    fpath = os.path.join(DATA_PATH, fname)
    with h5py.File(fpath, 'r') as f:
        # Take the first dataset in file
        key = list(f.keys())[0]
        seg = np.array(f[key]).flatten()
        np.save(os.path.join(OUT_PATH, fname.replace(".mat", ".npy")), seg)
