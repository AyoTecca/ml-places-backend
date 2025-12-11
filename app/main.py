import os
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.search_service import (
    search,
    get_place_details,
    explain_place
)
from app.models.request_models import SearchRequest, ExplainRequest
from sentence_transformers import SentenceTransformer
from app.utils.loader import load_resources

_logger = logging.getLogger("uvicorn")
app = FastAPI(title="ML Places Backend")

_model, EMBEDDINGS, PLACES = load_resources()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

LAST_USER_QUERY = {}  


@app.post("/search")
async def api_search(req: SearchRequest):
    start = time.perf_counter()
    LAST_USER_QUERY["query"] = req.query  

    query_emb = embedding_model.encode(req.query, show_progress_bar=False)
    results = search(query_emb, top_k=req.top_k)

    return {
        "time_ms": round((time.perf_counter() - start) * 1000, 2),
        "results": results
    }


@app.get("/place_details/{place_id}")
async def api_place_details(place_id: str, include_reviews: bool = True):
    try:
        details = get_place_details(place_id, include_reviews=include_reviews, include_nearby_full=True)
    except KeyError:
        raise HTTPException(status_code=404, detail="place not found")

    return details


@app.post("/explain_place")
async def api_explain_place(req: ExplainRequest):
    t0 = time.perf_counter()

    user_query = LAST_USER_QUERY.get("query", "")
    if not user_query:
        raise HTTPException(status_code=400, detail="No stored query available")

    result = await explain_place(req.place_id, user_query)
    return {
        "place_id": req.place_id,
        "explanation": result["ai_explanation"],
        "explain_ms": result["explain_ms"],
        "total_ms": round((time.perf_counter() - t0) * 1000, 2)
    }
