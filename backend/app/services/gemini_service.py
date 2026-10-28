import base64
import logging
from typing import List, Dict, Any, Optional
import httpx
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Google Gemini Client.
    Supports chat, multimodal vision, and embedding generation via Google GenAI REST API.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.default_model = settings.GEMINI_MODEL

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate_chat_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[str]] = None,
    ) -> str:
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured.")

        selected_model = model or self.default_model
        endpoint = f"{self.base_url}/models/{selected_model}:generateContent?key={self.api_key}"

        contents = []

        # Add conversation history
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        # Augment prompt with context if RAG or Web Search chunks are present
        final_prompt = prompt
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            final_prompt = (
                f"You have access to the following relevant context:\n\n"
                f"{context_text}\n\n"
                f"Instructions:\n"
                f"1. Synthesize your answer using the provided context while addressing the user's inquiry thoroughly.\n"
                f"2. Clearly present facts, citing relevant titles and web URLs where appropriate.\n"
                f"3. If answering a conceptual or general question, combine the context with foundational knowledge.\n\n"
                f"Question: {prompt}"
            )

        contents.append({"role": "user", "parts": [{"text": final_prompt}]})

        payload: Dict[str, Any] = {"contents": contents}

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        primary_model = model or self.default_model
        candidate_models = [primary_model]
        for fallback in ["gemini-3.1-flash-lite", "gemini-3.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for current_model in candidate_models:
            endpoint = f"{self.base_url}/models/{current_model}:generateContent?key={self.api_key}"
            try:
                with httpx.Client(timeout=httpx.Timeout(35.0, connect=3.0)) as client:
                    response = client.post(endpoint, json=payload)
                    if response.status_code in [429, 503, 500, 502, 504]:
                        last_error = f"Gemini API high demand on {current_model} (HTTP {response.status_code})."
                        continue
                    if response.status_code == 404:
                        last_error = f"Gemini model {current_model} not found."
                        continue
                    if response.status_code != 200:
                        last_error = f"Gemini API error on {current_model} {response.status_code}: {response.text}"
                        continue

                    data = response.json()
                    candidate = data["candidates"][0]
                    return candidate["content"]["parts"][0]["text"]
            except Exception as e:
                last_error = str(e)
                continue

        raise RuntimeError(f"All Gemini model attempts failed. Last error: {last_error}")

    def analyze_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        model: Optional[str] = None
    ) -> str:
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured.")

        base64_img = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt or "Describe and analyze this image in detail."},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_img
                            }
                        }
                    ]
                }
            ]
        }

        primary_model = model or self.default_model
        candidate_models = [primary_model]
        for fallback in ["gemini-3.1-flash-lite", "gemini-3.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for current_model in candidate_models:
            endpoint = f"{self.base_url}/models/{current_model}:generateContent?key={self.api_key}"
            try:
                with httpx.Client(timeout=httpx.Timeout(45.0, connect=3.0)) as client:
                    response = client.post(endpoint, json=payload)
                    if response.status_code in [429, 503, 500, 502, 504]:
                        last_error = f"Gemini Vision rate limit/demand on {current_model} (HTTP {response.status_code})."
                        continue
                    if response.status_code != 200:
                        last_error = f"Gemini Vision error on {current_model} {response.status_code}: {response.text}"
                        continue

                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                last_error = str(e)
                continue

        raise RuntimeError(f"All Gemini vision model attempts failed: {last_error}")

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        if not self.is_configured():
            raise ValueError("GEMINI_API_KEY is not configured.")

        embed_model = model or settings.GEMINI_EMBEDDING_MODEL
        endpoint = f"{self.base_url}/models/{embed_model}:embedContent?key={self.api_key}"

        payload = {
            "model": f"models/{embed_model}",
            "content": {"parts": [{"text": text}]}
        }

        with httpx.Client(timeout=httpx.Timeout(30.0, connect=3.0)) as client:
            response = client.post(endpoint, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Gemini Embedding error {response.status_code}: {response.text}")
            data = response.json()
            return data["embedding"]["values"]
