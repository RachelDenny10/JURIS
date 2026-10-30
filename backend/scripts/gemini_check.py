"""
Gemini connectivity check (never prints the API key).

  python scripts\gemini_check.py                 list models your key can call
  python scripts\gemini_check.py --all           include image/audio/live models too
  python scripts\gemini_check.py --probe NAME    send one tiny test request to model NAME

Run from the backend folder: the key is read from backend\.env via the app settings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from app.core.config import get_settings  # noqa: E402

BASE = "https://generativelanguage.googleapis.com/v1beta"
SKIP_WORDS = ("image", "tts", "audio", "live", "embedding", "robotics", "computer-use", "veo", "imagen")


def _redact(text: str, key: str) -> str:
    return text.replace(key, "<redacted>") if key else text


def _explain_error(response: requests.Response, key: str) -> str:
    try:
        error = response.json().get("error", {})
        message = str(error.get("message", ""))[:300]
        code = error.get("status", "")
    except ValueError:
        message, code = response.text[:300], ""
    return _redact(f"HTTP {response.status_code} {code}: {message}", key)


def list_models(key: str, show_all: bool) -> None:
    models, token = [], None
    while True:
        params = {"pageSize": 100}
        if token:
            params["pageToken"] = token
        try:
            response = requests.get(f"{BASE}/models", headers={"x-goog-api-key": key}, params=params, timeout=30)
        except requests.RequestException as exc:
            print("Network problem:", type(exc).__name__)
            sys.exit(1)
        if not response.ok:
            print("Google rejected the request ->", _explain_error(response, key))
            sys.exit(1)
        data = response.json()
        models.extend(data.get("models", []))
        token = data.get("nextPageToken")
        if not token:
            break

    usable = [m for m in models if "generateContent" in m.get("supportedGenerationMethods", [])]
    if not show_all:
        usable = [m for m in usable if not any(w in m["name"].lower() for w in SKIP_WORDS)]

    print(f"Key accepted. {len(models)} models listed, {len(usable)} shown (generateContent):\n")
    print(f"{'model name (use this in LLM_MODEL)':<46}{'in-tokens':>10}{'out-tokens':>11}")
    for m in sorted(usable, key=lambda m: m["name"]):
        name = m["name"].removeprefix("models/")
        print(f"{name:<46}{m.get('inputTokenLimit', '?'):>10}{m.get('outputTokenLimit', '?'):>11}")
    print("\nA model being listed does not prove it works for your project; use --probe NAME to test one.")


def probe(key: str, model: str) -> None:
    body = {
        "contents": [{"parts": [{"text": "Reply with the single word OK."}]}],
        "generationConfig": {"maxOutputTokens": 256, "temperature": 0},
    }
    try:
        response = requests.post(
            f"{BASE}/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
    except requests.RequestException as exc:
        print("Network problem:", type(exc).__name__)
        sys.exit(1)
    if not response.ok:
        print(f"Model '{model}' failed ->", _explain_error(response, key))
        sys.exit(1)

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        print("No candidates returned. promptFeedback:", data.get("promptFeedback"))
        sys.exit(1)
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    usage = data.get("usageMetadata", {})
    print(f"Model '{model}' answered.")
    print("finishReason     :", candidates[0].get("finishReason"))
    print("reply text       :", repr(text[:100]))
    print("prompt tokens    :", usage.get("promptTokenCount"))
    print("answer tokens    :", usage.get("candidatesTokenCount"))
    print("thinking tokens  :", usage.get("thoughtsTokenCount"))


def main() -> None:
    key = get_settings().gemini_api_key.strip()
    if not key:
        print("GEMINI_API_KEY is not set in backend\\.env")
        sys.exit(1)
    args = sys.argv[1:]
    if "--probe" in args:
        index = args.index("--probe")
        if index + 1 >= len(args):
            print("Usage: --probe MODEL_NAME")
            sys.exit(1)
        probe(key, args[index + 1])
    else:
        list_models(key, "--all" in args)


if __name__ == "__main__":
    main()