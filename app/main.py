from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.request_models import SearchRequest
from app.services.search_service import search_places
from app.utils.loader import load_resources

app = FastAPI(title="Places Semantic Search API")

# Load model and embeddings once at startup
model, embeddings, places_index = load_resources()

# Enable CORS
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
