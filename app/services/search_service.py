import numpy as np
import re

try:
    from transliterate import translit
    HAS_TRANSLIT = True
except:
    HAS_TRANSLIT = False


def clean_address(addr: str, transliterate: bool = False) -> str:
    if not isinstance(addr, str) or addr.lower() == "unknown_address":
        return "Address not available"   # <-- Fix #1

    addr = re.sub(r"[\uE000-\uF8FF]", "", addr)

    addr = re.sub(r"\b[A-Z0-9]{4,8}\+[A-Z0-9]{2,3}\b,?", "", addr)

    addr = re.sub(r"^[абвгдежзийклмнопрстуфхцчшщъыьэюя]\s*,\s*", "", addr, flags=re.I)

    parts = [p.strip() for p in addr.split(",") if p.strip()]

    parts = [re.sub(r"\b050\d{2}\b", "", p).strip() for p in parts]

    parts = list(dict.fromkeys(parts))

    addr = ", ".join(parts)

    addr = re.sub(r"\s+", " ", addr).strip(" ,")

    if transliterate and HAS_TRANSLIT:
        addr = translit(addr, "ru", reversed=True)

    return addr or "Address not available"

def search_places(req, model, embeddings, places_index):
    query_emb = model.encode([req.query])[0]

    sims = embeddings @ query_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )

    top_k = req.top_k
    idxs = np.argsort(sims)[-top_k:][::-1]

    results = []

    for idx in idxs:
        row = places_index.iloc[idx]

        address = clean_address(row.get("address"))

        results.append({
            "place_id": row["place_id"],
            "name": row["place_name"],
            "address": address,
            "category": row.get("auto_category"),
            "rating": float(row["rating_norm"]),
            "lat": row["lat"],
            "lng": row["lng"],
            "similarity": float(sims[idx]),
        })

    return {"results": results}
