"""
LLM access layer (Phase 13).

Every call to an external language model goes through this module, so
the rest of JURIS never touches API keys or provider details.

Providers (LLM_PROVIDER in .env):
  none       no LLM; callers fall back to extractive behaviour
  gemini     Google Gemini API (generateContent)
  openai     OpenAI Chat Completions API
  anthropic  Anthropic Messages API

Plain HTTPS via `requests` (installed as a spaCy dependency), so no
provider SDK is needed. API keys come only from settings/.env, are sent
in request headers (never in a URL), and are never logged or returned;
any provider error text is redacted before it is used.
"""

import logging
import re
import time

import requests

from app.core.config import get_settings

logger = logging.getLogger("juris.llm_service")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Model names change and retire quickly (gemini-2.5-flash already returns 404 for new
# users). Set LLM_MODEL in .env explicitly and verify it with scripts\gemini_check.py.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
SUPPORTED_PROVIDERS = {"none", "gemini", "openai", "anthropic"}
KEY_ENV_NAMES = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._\-]+$")


class LLMNotConfiguredError(Exception):
    """Raised when a provider is selected but cannot be used (missing key, unknown provider)."""


class LLMError(Exception):
    """Raised when a request to the provider fails (network, auth, quota, bad response)."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def _provider() -> str:
    return (get_settings().llm_provider or "none").strip().lower()


def _api_key(provider: str) -> str:
    settings = get_settings()
    keys = {
        "gemini": settings.gemini_api_key,
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
    }
    return (keys.get(provider) or "").strip()


def _model_name(provider: str) -> str:
    configured = get_settings().llm_model.strip()
    if configured.startswith("models/"):
        configured = configured[len("models/"):]
    return configured or DEFAULT_MODELS[provider]


def provider_status() -> dict:
    """Describe the current LLM setup without exposing any key."""
    provider = _provider()
    if provider == "none":
        return {
            "provider": "none",
            "llm_model": None,
            "configured": False,
            "detail": "No LLM selected (LLM_PROVIDER=none): summaries are extractive.",
        }
    if provider not in SUPPORTED_PROVIDERS:
        return {
            "provider": provider,
            "llm_model": None,
            "configured": False,
            "detail": f"Unsupported LLM_PROVIDER '{provider}'. Use none, gemini, openai or anthropic.",
        }

    model = _model_name(provider)
    if not _MODEL_NAME_PATTERN.match(model):
        return {
            "provider": provider,
            "llm_model": None,
            "configured": False,
            "detail": "LLM_MODEL contains unsupported characters.",
        }
    if not _api_key(provider):
        return {
            "provider": provider,
            "llm_model": model,
            "configured": False,
            "detail": f"{KEY_ENV_NAMES[provider]} is not set in backend/.env.",
        }
    return {"provider": provider, "llm_model": model, "configured": True, "detail": "Ready."}


# --------------------------------------------------------------------------
# Error handling helpers
# --------------------------------------------------------------------------

def _redact(text: str, key: str) -> str:
    return text.replace(key, "<redacted>") if key else text


def _provider_error_message(response: requests.Response, key: str) -> str:
    """Pull a short, key-free error message out of a provider's error body."""
    try:
        error = response.json().get("error")
        if isinstance(error, dict):
            message = str(error.get("message", ""))
        elif error:
            message = str(error)
        else:
            message = response.text
    except ValueError:
        message = response.text
    return _redact(message, key)[:300]


def _raise_for_status(response: requests.Response, provider: str, model: str, key: str) -> None:
    """Translate HTTP failures into short, key-free LLMError messages."""
    if response.ok:
        return

    status = response.status_code
    message = _provider_error_message(response, key)

    # Google reports an invalid key as HTTP 400 ("API key not valid"), not 401.
    key_rejected = status in (401, 403) or (status == 400 and "api key" in message.lower())
    if key_rejected:
        raise LLMError(
            f"{provider} rejected the API key (HTTP {status}). "
            f"Check {KEY_ENV_NAMES[provider]} in backend/.env."
        )
    if status == 404:
        raise LLMError(
            f"{provider} model '{model}' is not available to this key (HTTP 404: {message}). "
            "Run scripts\\gemini_check.py to list models and set LLM_MODEL."
        )
    if status == 429:
        raise LLMError(f"{provider} rate limit or quota exceeded (HTTP 429). Check usage limits and billing.")
    if status >= 500:
        raise LLMError(f"{provider} is temporarily unavailable (HTTP {status}).")
    raise LLMError(f"{provider} rejected the request (HTTP {status}): {message}")


# --------------------------------------------------------------------------
# Request building and response parsing, per provider
# --------------------------------------------------------------------------

def _build_request(provider: str, model: str, key: str, system_prompt: str, user_prompt: str):
    settings = get_settings()

    if provider == "gemini":
        url = f"{GEMINI_BASE_URL}/{model}:generateContent"
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": settings.llm_max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
    elif provider == "openai":
        url = OPENAI_URL
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": settings.llm_max_output_tokens,
            "response_format": {"type": "json_object"},
        }
    else:  # anthropic
        url = ANTHROPIC_URL
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": settings.llm_max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    return url, headers, payload


def _extract_text(provider: str, data: dict) -> str:
    """Pull the reply text out of a provider's JSON response."""
    if provider == "gemini":
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(
                "gemini returned no answer" + (f" (blocked: {block})." if block else ".")
            )
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        parts = (candidate.get("content") or {}).get("parts") or []
        # Skip any "thought" parts; only the model's actual answer is wanted.
        text = "".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")
        )
        if finish == "MAX_TOKENS":
            raise LLMError(
                "gemini stopped because the output limit was reached, so the reply is incomplete. "
                "Raise LLM_MAX_OUTPUT_TOKENS in backend/.env."
            )
        if finish not in (None, "STOP") and not text.strip():
            raise LLMError(f"gemini stopped without an answer (reason: {finish}).")
        return text

    if provider == "openai":
        return data["choices"][0]["message"]["content"]

    return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
RETRY_STATUSES = {500, 502, 503, 504}
RETRY_DELAYS_SECONDS = (3, 8)  # waits before the 2nd and 3rd attempt


def _post_with_retry(url, headers, payload, provider: str, timeout: int) -> requests.Response:
    """POST to the provider, retrying briefly on transient server errors (5xx only)."""
    attempts = len(RETRY_DELAYS_SECONDS) + 1
    response = None
    for attempt in range(attempts):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.Timeout as exc:
            raise LLMError(f"The {provider} request timed out after {timeout}s.") from exc
        except requests.RequestException:
            raise LLMError(f"Could not reach the {provider} API (network problem).") from None

        if response.status_code not in RETRY_STATUSES or attempt == attempts - 1:
            return response

        delay = RETRY_DELAYS_SECONDS[attempt]
        logger.warning(
            "%s returned HTTP %d; retrying in %ds (attempt %d of %d)",
            provider, response.status_code, delay, attempt + 2, attempts,
        )
        time.sleep(delay)
    return response


def generate(system_prompt: str, user_prompt: str) -> str:
    """Send one prompt to the configured provider and return the reply text."""
    status = provider_status()
    if not status["configured"]:
        raise LLMNotConfiguredError(status["detail"])

    provider = status["provider"]
    model = status["llm_model"]
    settings = get_settings()
    key = _api_key(provider)

    url, headers, payload = _build_request(provider, model, key, system_prompt, user_prompt)

    logger.info(
        "LLM request: provider=%s model=%s prompt_chars=%d",
        provider, model, len(system_prompt) + len(user_prompt),
    )

    response = _post_with_retry(url, headers, payload, provider, settings.llm_timeout_seconds)
    _raise_for_status(response, provider, model, key)

    try:
        text = _extract_text(provider, response.json())
    except LLMError:
        raise
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError(f"Unexpected response format from {provider}.") from exc

    if not text or not text.strip():
        raise LLMError(f"{provider} returned an empty response.")
    return text