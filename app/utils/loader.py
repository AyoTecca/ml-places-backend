import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_resources():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading DataFrame pickle...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(base_dir, "data", "places_index_with_reviews.pkl"),
        os.path.join(base_dir, "data", "places_index_with_categories.pkl"),
        os.path.join(base_dir, "data", "places_index.pkl"),
    ]

    pickle_path = None
    for c in candidates:
        if os.path.exists(c):
            pickle_path = c
            break

    if not pickle_path:
        raise FileNotFoundError("No places_index pickle found. Tried: " + ", ".join(candidates))

    print("Looking for pickle at:", pickle_path)

    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    print("Pickle loaded. Type:", type(df))

    embeddings = np.vstack(df["embedding"].values)
    places_index = df.drop(columns=["embedding"])

    print("+ Loaded embeddings:", embeddings.shape)
    print("+ Loaded index:", len(places_index))

    return model, embeddings, places_index
