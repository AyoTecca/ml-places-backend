import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_places(req, model, embeddings, places_index):

    query_embed = model.encode(req.query)

    sims = cosine_similarity([query_embed], embeddings)[0]

    idx = np.argsort(sims)[::-1][:req.top_k]

    results = []
    for i in idx:
        row = places_index.iloc[i]

        results.append({
            "place_id": row["place_id"],
            "name": row["place_name"],     
            "address": row.get("address", None),  
            "categories": row.get("categories", None),
            "rating": row.get("rating_norm", None),
            "lat": float(row.get("lat", 0)),
            "lng": float(row.get("lng", 0)),
            "similarity": float(sims[i]),
        })

    return {"results": results}
