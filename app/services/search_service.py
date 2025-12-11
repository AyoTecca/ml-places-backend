import numpy as np
import re
import math
from typing import Optional, List, Dict
from app.services.llm_service import run_llm   

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from transliterate import translit
    HAS_TRANSLIT = True
except Exception:
    HAS_TRANSLIT = False



def clean_address(addr: str, transliterate: bool = False) -> Optional[str]:
    if not isinstance(addr, str) or addr.lower() == "unknown_address":
        return "Address unavailable"

    addr = re.sub(r"[\uE000-\uF8FF]", "", addr)
    addr = re.sub(r"\b[A-Z0-9]{4,8}\+[A-Z0-9]{2,3}\b,?", "", addr)
    addr = re.sub(r"^[абвгдежзийклмнопрстуфхцчшщъыьэюя]\s*,\s*", "", addr, flags=re.I)

    parts = [p.strip() for p in addr.split(",") if p.strip()]
    parts = [re.sub(r"\b050\d{2}\b", "", p).strip() for p in parts]
    parts = [p for p in parts if p]
    parts = list(dict.fromkeys(parts))

    addr = ", ".join(parts)
    addr = re.sub(r"\s+", " ", addr).strip(" ,")

    if transliterate and HAS_TRANSLIT:
        try:
            addr = translit(addr, "ru", reversed=True)
        except Exception:
            pass

    return addr or None



_SENT_PIPELINE = None

def prepare_sentiment_pipeline(force_reload: bool = False):
    global _SENT_PIPELINE
    if _SENT_PIPELINE is not None and not force_reload:
        return _SENT_PIPELINE

    if not HAS_TRANSFORMERS:
        _SENT_PIPELINE = None
        return None

    try:
        _SENT_PIPELINE = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception:
        try:
            _SENT_PIPELINE = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        except Exception:
            _SENT_PIPELINE = None

    return _SENT_PIPELINE


def score_texts_sentiment(texts: List[str]) -> Optional[float]:
    if not texts:
        return None

    pipe = prepare_sentiment_pipeline()
    if pipe is None:
        return None

    try:
        out = pipe(texts, truncation=True)
    except Exception:
        return None

    scores = []
    for o in out:
        if isinstance(o, dict) and "label" in o:
            label = o["label"]
            m = re.search(r"(\d)", str(label))
            if m:
                val = int(m.group(1))
                scores.append((val - 1) / 4.0)
            elif "POSITIVE" in label.upper():
                scores.append(float(o.get("score", 1.0)))
            elif "NEGATIVE" in label.upper():
                scores.append(1.0 - float(o.get("score", 1.0)))
            else:
                scores.append(float(o.get("score", 0.5)))
        else:
            scores.append(float(o.get("score", 0.5)))

    if not scores:
        return None

    return float(sum(scores) / len(scores))



def explain_place(row: Dict, similarity: float, sentiment: Optional[float], combined_score: float) -> str:
    name = row.get("place_name", "This place")
    rating = float(row.get("rating_norm", row.get("avg_rating", 0)))
    category = row.get("auto_category", "place")
    reasons = []

    if rating >= 0.9:
        reasons.append("strong official rating")
    elif rating >= 0.75:
        reasons.append("good official rating")
    else:
        reasons.append("moderate official rating")

    if sentiment is not None:
        if sentiment >= 0.8:
            reasons.append("very positive reviews")
        elif sentiment >= 0.6:
            reasons.append("mostly positive reviews")
        else:
            reasons.append("mixed/negative reviews")
    else:
        reasons.append("insufficient sentiment data")

    if similarity >= 0.6:
        reasons.append("very strong semantic match to the query")
    elif similarity >= 0.45:
        reasons.append("good semantic match")

    if category:
        reasons.append(f"category: {category}")

    explanation = f"{name} — recommended because of " + ", ".join(reasons) + f". Combined score: {combined_score:.2f}."
    return explanation



def combined_score(similarity: float, sentiment: Optional[float], rating_norm: float) -> float:
    sim_w = 0.5
    sent_w = 0.3 if sentiment is not None else 0.0
    rating_w = 0.2 if sentiment is not None else 0.5
    total_w = sim_w + sent_w + rating_w
    if total_w == 0:
        return similarity
    return max(0.0, min(1.0, (sim_w * similarity + sent_w * (sentiment or 0) + rating_w * rating_norm) / total_w))



def search_places(req, model, embeddings, places_index):
    query_emb = model.encode([req.query])[0]
    sims = embeddings @ query_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    top_k = req.top_k
    idxs = np.argsort(sims)[-top_k:][::-1]

    results = []
    for idx in idxs:
        row = places_index.iloc[idx]
        address = clean_address(
            row.get("address"),
            transliterate=getattr(req, "transliterate", False)
        )
        results.append({
            "place_id": row["place_id"],
            "name": row["place_name"],
            "address": address,
            "category": row.get("auto_category"),
            "rating": float(row.get("rating_norm", row.get("avg_rating", 0))),
            "lat": row["lat"],
            "lng": row["lng"],
            "similarity": float(sims[idx]),
        })
    return {"results": results}



def search_places_extended(req, model, embeddings, places_index):

    query_emb = model.encode([req.query])[0]
    sims = embeddings @ query_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    idxs_by_sim = np.argsort(sims)[-req.top_k:][::-1]
    candidates = []

    prepare_sentiment_pipeline()

    for idx in idxs_by_sim:
        row = places_index.iloc[idx].to_dict()

        addr = clean_address(
            row.get("address"),
            transliterate=getattr(req, "transliterate", False)
        )

        sentiment_score = None
        if "reviews" in places_index.columns:
            cell = row.get("reviews")
            texts = cell[:8] if isinstance(cell, list) else ([cell] if isinstance(cell, str) else [])
            if texts:
                sentiment_score = score_texts_sentiment(texts)

        rating_norm = float(row.get("rating_norm", row.get("avg_rating", 0) or 0))

        if sentiment_score is None:
            sentiment_score = rating_norm

        sim_val = float(sims[idx])
        combined = combined_score(sim_val, sentiment_score, rating_norm)

        if getattr(req, "use_llm_explanations", False):
            review_text = "\n".join(texts) if texts else ""
            prompt = f"""
You are an assistant helping users choose places in Almaty.

User query: "{req.query}"

Place: {row.get("place_name")}
Category: {row.get("auto_category")}
Address: {addr}

Similarity to query: {sim_val:.2f}
Official rating: {rating_norm:.2f}
Review sentiment score: {sentiment_score:.2f}

Reviews:
{review_text}

Write a clear, helpful explanation (3–4 sentences) describing:
- why this place matches the user query  
- whether sentiment matches the official rating  
- any concerns or warnings users should know  
- whether this place is a good recommendation  
"""
            explanation = run_llm(prompt) or explain_place(row, sim_val, sentiment_score, combined)

        else:
            explanation = explain_place(row, sim_val, sentiment_score, combined)

        candidates.append({
            "place_id": row.get("place_id"),
            "name": row.get("place_name"),
            "address": addr,
            "category": row.get("auto_category"),
            "rating": rating_norm,
            "sentiment": sentiment_score,
            "similarity": sim_val,
            "combined_score": combined,
            "ai_explanation": explanation,
            "lat": row.get("lat"),
            "lng": row.get("lng"),
        })

    candidates.sort(key=lambda x: x["combined_score"], reverse=True)

    avg_similarity = float(np.mean([c["similarity"] for c in candidates]))
    avg_rating = float(np.mean([c["rating"] for c in candidates]))
    avg_sentiment = float(np.mean([c["sentiment"] for c in candidates]))

    rating_vs_sentiment_note = (
        "ratings and reviews match well."
        if abs(avg_rating - avg_sentiment) < 0.05 else
        "ratings appear higher than sentiment."
        if avg_rating > avg_sentiment else
        "reviews are more positive than ratings."
    )

    # =====================================================
    #            LLM QUERY SUMMARY (NEW LOGIC)
    # =====================================================
    query_summary = {
        "user_intent": req.query,
        "avg_similarity": avg_similarity,
        "avg_rating": avg_rating,
        "avg_sentiment": avg_sentiment,
        "rating_vs_sentiment_note": rating_vs_sentiment_note,
        "ai_comment": f"Top {len(candidates)} places identified based on rating, sentiment, and embedding similarity."
    }

    if getattr(req, "use_llm_explanations", False):
        summary_prompt = f"""
The user searched for: "{req.query}"

We recommended {len(candidates)} places.
Average semantic similarity: {avg_similarity:.2f}
Average rating: {avg_rating:.2f}
Average sentiment: {avg_sentiment:.2f}

Write a short summary (3–5 sentences):
- interpret the user's intent  
- describe overall quality of recommendations  
- highlight category patterns  
- explain if ratings disagree with sentiment  
- give final advice to the user  
"""
        llm_summary = run_llm(summary_prompt)
        if llm_summary:
            query_summary["ai_comment"] = llm_summary

    return {"query_summary": query_summary, "results": candidates}
