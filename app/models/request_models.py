from pydantic import BaseModel
from typing import Optional

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    transliterate: Optional[bool] = False


class ExplainRequest(BaseModel):
    place_id: str