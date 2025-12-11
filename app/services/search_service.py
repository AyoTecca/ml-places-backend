import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.loader import load_resources
from app.services.llm_service import explain_place_async

_logger = logging.getLogger("search_service")
_model, EMBEDDINGS, PLACES = load_resources()


def safe_get_address(r):
    a = r.get("address")
    return a if isinstance(a, str) and a.strip() else "Address not available"


def safe_get_category(r):
    for c in ("auto_category", "category"):
        if isinstance(r.get(c), str) and r[c].strip():
            return r[c]
    return None


def search(query_emb, top_k=10):
    sims = cosine_similarity(query_emb.reshape(1, -1), EMBEDDINGS).reshape(-1)
    idx = np.argsort(-sims)[:top_k]

    out = []
    for i in idx:
        r = PLACES.iloc[i].to_dict()
        out.append({
            "place_id": r["place_id"],
            "name": r["place_name"],
            "address": safe_get_address(r),
            "category": safe_get_category(r),
            "rating": float(r.get("rating_norm") or r.get("avg_rating") or 0.0),
            "similarity": float(sims[i]),
            "lat": r["lat"],
            "lng": r["lng"]
        })
    return out


def get_place_details(place_id: str, include_reviews=True, include_nearby_full=True):
    row = PLACES[PLACES.place_id == place_id]
    if len(row) == 0:
        raise KeyError(place_id)

    r = row.iloc[0].to_dict()

    details = {
        "place_id": r["place_id"],
        "name": r["place_name"],
        "address": safe_get_address(r),
        "category": safe_get_category(r),
        "rating": float(r.get("rating_norm") or r.get("avg_rating") or 0.0),
        "lat": r["lat"],
        "lng": r["lng"],
        "sentiment": float(r.get("sentiment") or 0.0),
        "reviews_count": int(r.get("review_count") or 0),
        "raw_review_snippets": r.get("review_snippets") or r.get("reviews") or None
    }

    if include_nearby_full:
        emb_arr = np.vstack(PLACES["embedding"].values)
        idx = int(row.index[0])
        sims = cosine_similarity(emb_arr[idx].reshape(1, -1), emb_arr).reshape(-1)

        sorted_idx = np.argsort(-sims)[1:6]
        nearby = []
        for j in sorted_idx:
            pr = PLACES.iloc[j].to_dict()
            nearby.append({
                "place_id": pr["place_id"],
                "name": pr["place_name"],
                "address": safe_get_address(pr),
                "category": safe_get_category(pr),
                "rating": float(pr.get("rating_norm") or pr.get("avg_rating") or 0.0),
                "sentiment": float(pr.get("sentiment") or 0.0),
                "lat": pr["lat"],
                "lng": pr["lng"],
                "similarity": float(sims[j])
            })

        details["nearby"] = nearby

    return details


async def explain_place(place_id: str, user_query: str):
    details = get_place_details(place_id, include_reviews=True)
    explanation, ms = await explain_place_async(details, user_query)
    return {
        "place_id": place_id,
        "ai_explanation": explanation,
        "explain_ms": ms
    }
