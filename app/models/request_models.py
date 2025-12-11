from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    transliterate: bool = False       # optional: transliterate Cyrillic addresses
    use_llm_explanations: bool = True  # if True, backend will attempt using an LLM for richer textual explanations (requires a configured LLM client)
