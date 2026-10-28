import base64
import logging
from typing import List, Dict, Any, Optional
import httpx
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class GrokService:
    """
    xAI Grok API Client.
    Connects to xAI's OpenAI-compatible endpoint at https://api.x.ai/v1.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GROK_API_KEY
        self.base_url = settings.GROK_BASE_URL.rstrip("/")
        self.default_model = settings.GROK_MODEL

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
            raise ValueError("GROK_API_KEY is not configured.")

        selected_model = model or self.default_model
        endpoint = f"{self.base_url}/chat/completions"

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are Grok, an intelligent, helpful, and insightful AI assistant."
            })

        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Grounding with RAG context if provided
        final_prompt = prompt
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            final_prompt = (
                f"Use ONLY the following retrieved context to answer the question. "
                f"If the answer cannot be found in the context, explicitly state that the answer is not available in the provided documents.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {prompt}"
            )

        messages.append({"role": "user", "content": final_prompt})

        payload = {
            "model": selected_model,
            "messages": messages,
            "temperature": 0.7
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=45.0) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            if response.status_code in [401, 402, 403, 429, 500, 502, 503]:
                raise RuntimeError(f"Grok API quota/limit exceeded (HTTP {response.status_code}).")
            if response.status_code != 200:
                raise RuntimeError(f"Grok API error {response.status_code}: {response.text}")

            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Failed to parse Grok response: {data}") from e

    def analyze_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        model: Optional[str] = None
    ) -> str:
        if not self.is_configured():
            raise ValueError("GROK_API_KEY is not configured.")

        selected_model = model or settings.GROK_VISION_MODEL
        endpoint = f"{self.base_url}/chat/completions"

        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{base64_img}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe and analyze this image in detail."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        payload = {
            "model": selected_model,
            "messages": messages
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            if response.status_code == 429:
                raise RuntimeError("Grok Vision rate limit exceeded (HTTP 429).")
            if response.status_code != 200:
                raise RuntimeError(f"Grok Vision error {response.status_code}: {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"]
