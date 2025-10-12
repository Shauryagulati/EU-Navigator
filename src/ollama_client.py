import json, os, requests, time
from typing import Optional
from requests.exceptions import ConnectionError, ReadTimeout

RETRIES = int(os.getenv("OLLAMA_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("OLLAMA_BACKOFF", "0.8"))

BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")

class OllamaError(RuntimeError): pass

def _collect_response_text(r: requests.Response) -> str:
    r.raise_for_status()
    text = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in obj:
            raise OllamaError(obj["error"])
        text += obj.get("response", "")
        if obj.get("done"):
            break
    return text.strip()

def ollama_generate(model: str, prompt: str, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
    payload = {"model": model, "prompt": prompt, "options": {"temperature": temperature}, "stream": True}
    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens
    url = f"{BASE}/api/generate"

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.post(url, json=payload, stream=True, timeout=600) as r:
                return _collect_response_text(r)
        except (ConnectionError, ReadTimeout) as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * attempt)
    raise OllamaError(f"Ollama request failed after {RETRIES} retries: {last_err}")
