from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.request_models import SearchRequest, ExplainPlaceRequest
from app.services.search_service import search, get_place_details, build_llm_explanation
from app.utils.loader import load_resources
from sentence_transformers import SentenceTransformer
import time

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model, EMBEDS, PLACES = load_resources()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    t0 = time.time()
    query_text = req.query
    if req.transliterate:
        query_text = query_text

    q_emb = embedder.encode(query_text)
    results = search(q_emb, top_k=req.top_k)
    return {
        "time_ms": round((time.time() - t0) * 1000, 2),
        "results": results
    }


@app.get("/place_details/{place_id}")
async def details_endpoint(place_id: str):
    t0 = time.time()
    try:
        d = get_place_details(place_id, include_reviews=True, include_nearby_full=True)
    except KeyError:
        raise HTTPException(status_code=404, detail="Place not found")
    d["details_ms"] = round((time.time() - t0) * 1000, 2)
    return d


@app.post("/explain_place")
async def explain_endpoint(req: ExplainPlaceRequest):
    t0 = time.time()

    d = get_place_details(req.place_id, include_reviews=True, include_nearby_full=True)
    enriched = await build_llm_explanation(d)

    out = {
        "place_id": req.place_id,
        "explanation": enriched["explanation"],
        "explain_ms": enriched["explain_ms"],
        "total_ms": round((time.time() - t0) * 1000, 2)
    }
    return out
