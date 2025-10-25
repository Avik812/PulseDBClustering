import numpy as np
import pandas as pd

def load_segments_csv(path, values_col='values'):
    """
    Expect CSV with columns: segment_id, values (string repr of list or JSON)
    """
    df = pd.read_csv(path)
    df[values_col] = df[values_col].apply(lambda s: np.array(eval(s)) if isinstance(s, str) else np.array(s))
    return df

def save_clusters(clusters, out_path):
    import json
    serial = {f'cluster_{i}': [int(x) for x in cl] for i, cl in enumerate(clusters)}
    with open(out_path, 'w') as f:
        json.dump(serial, f, indent=2)
