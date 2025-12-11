import os
import pickle
import numpy as np
import pandas as pd
import logging
_logger = logging.getLogger("loader")

APP_DIR = os.path.dirname(os.path.dirname(__file__))          
PROJECT_ROOT = os.path.dirname(APP_DIR)                       
DATA_DIRS = [
    os.path.join(APP_DIR, "data"),
    os.path.join(PROJECT_ROOT, "data"),
]

def find_pickle(preferred="places_index_with_reviews.pkl"):
    for folder in DATA_DIRS:
        p = os.path.join(folder, preferred)
        if os.path.exists(p):
            return p
    for folder in DATA_DIRS:
        for alt in ("places_index_with_categories.pkl", "places_index.pkl"):
            path = os.path.join(folder, alt)
            if os.path.exists(path):
                return path
    return None


def load_resources(pickle_name: str | None = None):
    model = None
    embeddings = None
    places_index = None
    if pickle_name is None:
        pickle_name = find_pickle()
    if pickle_name is None:
        raise FileNotFoundError("No places pickle found in data/")
    _logger.info(f"Looking for pickle at: {pickle_name}")
    with open(pickle_name, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded pickle is not a DataFrame")
    if "embedding" in df.columns:
        arrs = df["embedding"].values
        try:
            embeddings = np.vstack(arrs)
        except Exception:
            embeddings = np.array(list(arrs.tolist()))
    else:
        raise KeyError("No 'embedding' column in places index")
    places_index = df.reset_index(drop=True)
    _logger.info(f"Pickle loaded. Type: {type(places_index)}")
    _logger.info(f"+ Loaded embeddings: {embeddings.shape}")
    _logger.info(f"+ Loaded index: {len(places_index)}")
    return model, embeddings, places_index
