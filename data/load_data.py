import h5py
import numpy as np
import os

# TO VIEW DATASET DETAILS
# Path to your dataset folder
DATA_PATH = "/Users/avik/Documents/projects/PulseDBClustering/data/weinanwangrutgers/pulsedb-balanced-training-and-testing/versions/4"

def load_pulsedb_file(filename):
    path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(path):
        print(f"❌ File {filename} not found at {path}")
        return None

    with h5py.File(path, 'r') as f:
        print(f"\n=== Inspecting {filename} ===")
        keys = list(f.keys())
        print("Keys in file:", keys)

        # Read all datasets into numpy arrays (optional: just preview)
        for key in keys:
            data = np.array(f[key])
            print(f"{key} shape: {data.shape}")

    return f

if __name__ == "__main__":
    filenames = [
        "VitalDB_AAMI_Cal_Subset.mat",
        "VitalDB_AAMI_Test_Subset.mat",
        "VitalDB_CalBased_Test_Subset.mat",
        "VitalDB_CalFree_Test_Subset.mat",
        "VitalDB_Train_Subset.mat"
    ]

    for file in filenames:
        load_pulsedb_file(file)
