from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    transliterate: bool = False      
    use_llm_explanations: bool = True 
