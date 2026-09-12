"""Optional LLM client with two tiers (all OpenAI-compatible).

The fly is the retriever/index; LLMs only do language. Two tiers let you put
cheap always-on phrasing on a small local model and reserve deep synthesis for
an optional frontier model:

  small tier (always-on distillation + plan phrasing):
    FLY_LLM_URL      base URL, e.g. http://localhost:8080   (unset => disabled)
    FLY_LLM_MODEL    model name (default "local")
    FLY_LLM_KEY      bearer token, if the server needs one   (optional)
    FLY_LLM_TIMEOUT  seconds (default 20)

  frontier tier (optional periodic deep synthesis, e.g. the report):
    FLY_FRONTIER_URL     base URL, e.g. https://api.openai.com   (unset => disabled)
    FLY_FRONTIER_MODEL   model name (default "gpt-4o")
    FLY_FRONTIER_KEY     bearer token                             (optional)
    FLY_FRONTIER_TIMEOUT seconds (default 60)

`complete()` returns None on any problem (unset, unreachable, timeout, bad
response) so every caller degrades gracefully. Uses only the stdlib.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

# tier -> (url_env, model_env, key_env, timeout_env, default_model, default_timeout)
_TIERS = {
    "small": ("FLY_LLM_URL", "FLY_LLM_MODEL", "FLY_LLM_KEY", "FLY_LLM_TIMEOUT", "local", "20"),
    "frontier": ("FLY_FRONTIER_URL", "FLY_FRONTIER_MODEL", "FLY_FRONTIER_KEY",
                 "FLY_FRONTIER_TIMEOUT", "gpt-4o", "60"),
}


def llm_configured(tier: str = "small") -> bool:
    url_env = _TIERS[tier][0]
    return bool(os.environ.get(url_env))


def complete(
    prompt: str,
    system: str | None = None,
    max_tokens: int = 220,
    tier: str = "small",
) -> str | None:
    url_env, model_env, key_env, timeout_env, default_model, default_timeout = _TIERS[tier]
    base = os.environ.get(url_env)
    if not base:
        return None
    endpoint = base.rstrip("/") + "/v1/chat/completions"
    model = os.environ.get(model_env, default_model)
    timeout = float(os.environ.get(timeout_env, default_timeout))
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens,
         "temperature": 0.2, "stream": False}
    ).encode()
    req = urllib.request.Request(endpoint, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"]["content"]
        return text.strip() or None
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError):
        return None
