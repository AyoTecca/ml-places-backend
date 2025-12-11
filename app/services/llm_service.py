import os
import time
import asyncio
import logging
from typing import Tuple, Dict, Any

_logger = logging.getLogger("llm_service")

REMOTE_PROVIDER = os.environ.get("REMOTE_LLM_PROVIDER", "").lower()
REMOTE_API_KEY = os.environ.get("REMOTE_LLM_API_KEY", "")
REMOTE_ENABLED = bool(REMOTE_PROVIDER and REMOTE_API_KEY)

def _build_prompt(details: Dict[str, Any], user_query: str) -> str:
    lines = []
    lines.append(f"User query: {user_query}")
    lines.append(f"Place name: {details.get('name')}")
    lines.append(f"Category: {details.get('category') or details.get('auto_category')}")
    lines.append(f"Rating: {details.get('rating')}")
    if details.get("sentiment") is not None:
        lines.append(f"Sentiment: {details.get('sentiment')}")
    if details.get("raw_review_snippets"):
        lines.append("Top review snippets (raw):")
        snips = details.get("raw_review_snippets")
        if isinstance(snips, (list, tuple)):
            snips = snips[:4]
            for s in snips:
                lines.append(f"- {s}")
        else:
            for i, s in enumerate(str(snips).splitlines()):
                if i >= 4:
                    break
                lines.append(f"- {s.strip()}")
    return "\n".join(lines)

async def explain_place_async(details: Dict[str, Any], user_query: str) -> Tuple[str, float]:
    start = time.perf_counter()
    prompt = _build_prompt(details, user_query)

    if REMOTE_ENABLED:
        try:
            if REMOTE_PROVIDER == "gemini":
                from google import generativeai as genai
                genai.configure(api_key=REMOTE_API_KEY)
                def _call():
                    resp = genai.generate_text(model="models/text-bison-001", input=prompt)
                    return resp.text
                text = await asyncio.to_thread(_call)
                ms = (time.perf_counter() - start) * 1000.0
                return text, ms
            elif REMOTE_PROVIDER == "ollama":
                import requests
                url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
                payload = {"model": os.environ.get("OLLAMA_MODEL", "phi3:latest"), "prompt": prompt, "max_tokens": 300}
                headers = {"Authorization": f"Bearer {REMOTE_API_KEY}"} if REMOTE_API_KEY else {}
                def _call():
                    r = requests.post(url, json=payload, headers=headers, timeout=30)
                    r.raise_for_status()
                    j = r.json()
                    return j.get("completion", "") or j.get("text", "")
                text = await asyncio.to_thread(_call)
                ms = (time.perf_counter() - start) * 1000.0
                return text, ms
            else:
                _logger.warning("REMOTE_PROVIDER set but not implemented: %s", REMOTE_PROVIDER)
        except Exception as e:
            _logger.exception("Remote LLM call failed, falling back to local explanation: %s", e)

    text, gen_ms = await asyncio.to_thread(get_simple_explanation_sync, details, user_query)
    total_ms = (time.perf_counter() - start) * 1000.0
    return text, gen_ms

def get_simple_explanation_sync(details: Dict[str, Any], user_query: str) -> Tuple[str, float]:
    start = time.perf_counter()
    name = details.get("name", "Unknown")
    rating = details.get("rating", None)
    sentiment = details.get("sentiment", None)
    reviews = details.get("raw_review_snippets")
    parts = []
    parts.append(f"{name} matches the query '{user_query}'.")
    if rating is not None:
        parts.append(f"Official rating: {round(float(rating), 2)}.")
    if sentiment is not None:
        parts.append(f"Aggregated review sentiment: {round(float(sentiment),2)}.")
    try:
        if sentiment is not None and rating is not None:
            if float(rating) - float(sentiment) > 0.15:
                parts.append("Note: official rating is noticeably higher than sentiment derived from reviews; consider reading user comments.")
            elif float(sentiment) - float(rating) > 0.15:
                parts.append("Note: user sentiments are higher than the official rating, reviews suggest people enjoy it more than the rating suggests.")
    except Exception:
        pass
    if reviews:
        if isinstance(reviews, (list, tuple)):
            snips = reviews[:2]
        else:
            snips = [s.strip() for s in str(reviews).splitlines() if s.strip()][:2]
        if snips:
            parts.append("Example review snippets:")
            for s in snips:
                txt = s if len(s) <= 300 else s[:297] + "..."
                parts.append(f"- {txt}")
    parts.append("If you need a deeper analysis (tone, pros/cons, warnings), request a detailed explanation — this will call an LLM and may use external quota.")
    text = " ".join(parts)
    ms = (time.perf_counter() - start) * 1000.0
    return text, ms
