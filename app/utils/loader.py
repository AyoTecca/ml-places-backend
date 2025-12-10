import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_resources():

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading DataFrame pickle...")
    with open("data/places_index.pkl", "rb") as f:
        df = pickle.load(f)

    print("Pickle type:", type(df))

    if "embedding" not in df.columns:
        raise ValueError("- The DataFrame does not contain an 'embedding' column!")

    embeddings = np.vstack(df["embedding"].values)

    places_index = df.drop(columns=["embedding"])

    print("+ Loaded embeddings:", embeddings.shape)
    print("+ Loaded index:", len(places_index))

    return model, embeddings, places_index
