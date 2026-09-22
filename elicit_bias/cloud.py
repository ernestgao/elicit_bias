"""Explicit, single-attempt cloud transports; never load credential files."""
from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

from .common import EVALUATION_JUDGES


class CloudError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise CloudError(f"Missing environment variable: {name}")
    return value


def endpoint(name):
    value = required(name).rstrip("/")
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise CloudError(f"{name} is not a valid HTTPS base URL") from None
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise CloudError(f"{name} must be an HTTPS base URL without credentials, query or fragment")
    return value


class CloudClient:
    def __init__(self, backend, *, role="judge"):
        self.backend, self.role = backend, role
        if role == "judge" and backend not in EVALUATION_JUDGES:
            raise CloudError("Scoring supports only Azure and Gemini")
        if backend == "azure":
            self.base = endpoint("AZURE_OPENAI_ENDPOINT")
            self.key = required("AZURE_OPENAI_API_KEY")
            self.model = required("AZURE_OPENAI_DEPLOYMENT" if role == "judge" else "AZURE_TARGET_DEPLOYMENT")
        elif backend == "gemini":
            if role != "judge":
                raise CloudError("Gemini is implemented as a judge only")
            self.base = "https://generativelanguage.googleapis.com/v1beta"
            self.key, self.model = required("GEMINI_API_KEY"), required("GEMINI_MODEL")
        elif backend in ("mimo", "compatible"):
            prefix = "MIMO" if backend == "mimo" else "TARGET"
            self.base = endpoint(prefix + "_BASE_URL")
            self.key, self.model = required(prefix + "_API_KEY"), required(prefix + "_MODEL")
        else:
            raise CloudError("Unknown backend; no fallback is configured")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.model):
            raise CloudError("Model/deployment must be a plain identifier, not a URL or path")

    def complete(self, messages, *, temperature=0.0, max_tokens=None):
        import requests
        if self.backend == "gemini":
            url = f"{self.base}/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": self.key}
            config = {"temperature": 0.0, "responseMimeType": "application/json", "maxOutputTokens": 2048,
                      "thinkingConfig": {"thinkingBudget": 0}}
            payload = {"contents": [{"parts": [{"text": messages[0]["content"]}]}], "generationConfig": config}
        else:
            url = self.base + "/chat/completions"
            headers = {"api-key": self.key} if self.backend == "azure" else {"Authorization": "Bearer " + self.key}
            payload = {"model": self.model, "messages": messages}
            if self.backend == "azure":
                if self.role != "judge":
                    if self.model.lower().startswith("gpt-5"):
                        payload["reasoning_effort"] = "minimal"
                    else:
                        payload["temperature"] = temperature
                    if max_tokens is not None:
                        payload["max_completion_tokens"] = max_tokens
            else:
                payload.update(temperature=temperature, stream=False)
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
        try:
            # No implicit proxy/netrc credentials, redirects or transport retries.
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(url, headers=headers, json=payload, timeout=120, allow_redirects=False)
                if response.status_code != 200:
                    raise CloudError(f"{self.backend} HTTP status {response.status_code}", response.status_code)
                data = response.json()
            if self.backend == "gemini":
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise CloudError("Empty model response")
            return text, data.get("usage", data.get("usageMetadata", {}))
        except CloudError:
            raise
        except Exception:
            # Exceptions and response bodies can include service addresses or credentials.
            raise CloudError(f"{self.backend} transport or response error; details suppressed") from None
