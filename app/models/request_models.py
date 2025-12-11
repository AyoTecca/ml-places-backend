from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    transliterate: bool = False


class ExplainPlaceRequest(BaseModel):
    place_id: str
