import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_resources():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading DataFrame pickle...")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pickle_path = os.path.join(base_dir, "data", "places_index_with_categories.pkl")

    print("Looking for pickle at:", pickle_path)

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at: {pickle_path}")

    df = pickle.load(open(pickle_path, "rb"))

    print("Pickle loaded. Type:", type(df))

    embeddings = np.vstack(df["embedding"].values)
    places_index = df.drop(columns=["embedding"])

    print("+ Loaded embeddings:", embeddings.shape)
    print("+ Loaded index:", len(places_index))

    return model, embeddings, places_index
