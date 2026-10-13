"""Small local-only Ollama client for guest conversation."""

import json
import os
from typing import Dict, Iterator, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cancellation import CancellationToken


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_MAX_RESPONSE_TOKENS = 512


def stream_guest_answer(
    messages: List[Dict[str, str]],
    cancellation: CancellationToken | None = None,
) -> Iterator[str]:
    """Yield guest response text as local Ollama generates it."""
    base_url = (os.environ.get("OLLAMA_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    max_response_tokens = int(
        os.environ.get("OLLAMA_MAX_RESPONSE_TOKENS", DEFAULT_MAX_RESPONSE_TOKENS)
    )
    if max_response_tokens < 1:
        raise ValueError("OLLAMA_MAX_RESPONSE_TOKENS must be a positive integer.")
    request_body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": "10m",
        "options": {"temperature": 0.1, "num_predict": max_response_tokens},
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    received_content = False
    try:
        with urlopen(request, timeout=300) as response:
            if cancellation is not None:
                cancellation.attach_cancel_callback(response.close)
            for line in response:
                if cancellation is not None:
                    cancellation.check()
                if not line.strip():
                    continue
                payload = json.loads(line.decode("utf-8"))
                if payload.get("error"):
                    raise RuntimeError(f"Local Ollama error: {payload['error']}")
                content = payload.get("message", {}).get("content")
                if isinstance(content, str) and content:
                    received_content = True
                    yield content
            if cancellation is not None:
                cancellation.check()
    except HTTPError as error:
        if cancellation is not None:
            cancellation.check()
        raise RuntimeError(f"Local Ollama returned HTTP {error.code}.") from error
    except (URLError, TimeoutError) as error:
        if cancellation is not None:
            cancellation.check()
        raise RuntimeError(
            f"Local Ollama did not respond at {base_url}; check Ollama and the model."
        ) from error
    except (OSError, ValueError):
        if cancellation is not None:
            cancellation.check()
        raise
    finally:
        if cancellation is not None:
            cancellation.clear_cancel_callback()
    if not received_content:
        raise ValueError("Local Ollama returned an empty guest response.")


def answer_guest_prompt(messages: List[Dict[str, str]]) -> str:
    """Return a complete guest response when streaming is not needed."""
    return "".join(stream_guest_answer(messages)).strip()
