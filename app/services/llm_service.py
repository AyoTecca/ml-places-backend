import asyncio
import json
import time
import aiohttp

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"  


async def _ollama_generate(prompt: str) -> str:
    """Send prompt to Ollama asynchronously."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload) as resp:
            data = await resp.json()
            return data.get("response", "")


def build_prompt(place: dict) -> str:
    """Construct intelligent prompt using place details."""
    name = place.get("name")
    category = place.get("category")
    rating = place.get("rating")
    sentiment = place.get("sentiment")
    address = place.get("address")
    reviews = place.get("raw_review_snippets") or []
    nearby = place.get("nearby", [])

    review_text = "\n".join([f"- {r}" for r in reviews]) if reviews else "No reviews available."

    nearby_text = "\n".join([
        f"- {p['name']} ({p.get('category')}), rating {p.get('rating')}, sentiment {p.get('sentiment')}"
        for p in nearby
    ]) if nearby else "No nearby places found."

    return f"""
You are an expert location analyst.

Summarize the following place for a user deciding whether to visit it.

Place name: {name}
Category: {category}
Address: {address}

Official rating: {rating}
Review sentiment score: {sentiment}

Review snippets:
{review_text}

Nearby similar places:
{nearby_text}

Produce:
1. A short explanation of whether this place fits general user intentions (dates, hangout, family, etc.)
2. Pros & cons based ONLY on the data above.
3. Compare rating vs sentiment (if sentiment is lower → warn user)
4. Provide a risk/warning section if reviews imply issues.
5. Keep everything factual. No hallucinations.
6. Keep answer under 4-5 sentences.
"""


async def llm_explain_place(place_details: dict):
    """Return LLM text + duration."""
    t0 = time.time()
    prompt = build_prompt(place_details)
    text = await _ollama_generate(prompt)
    ms = (time.time() - t0) * 1000
    return text, ms