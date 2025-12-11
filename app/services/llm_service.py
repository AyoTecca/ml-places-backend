import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"  


def run_llm(prompt: str, max_tokens: int = 180) -> str:
    """
    Sends a prompt to the local Ollama LLM and returns generated text.
    Handles errors gracefully.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": False
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except Exception as e:
        return f"[LLM_ERROR: {e}]"
