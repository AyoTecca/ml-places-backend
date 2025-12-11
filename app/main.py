# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.request_models import SearchRequest
from app.services.search_service import search_places, search_places_extended
from app.utils.loader import load_resources

app = FastAPI(title="Places Semantic Search API (extended)")

model, embeddings, places_index = load_resources()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/search")
def search_endpoint(req: SearchRequest):
    return search_places(req, model, embeddings, places_index)

@app.post("/search_extended")
def search_extended_endpoint(req: SearchRequest):
    """
    Extended endpoint: returns query-level summary + per-result sentiment,
    combined score and an AI explanation for each recommended place.
    """
    return search_places_extended(req, model, embeddings, places_index)
