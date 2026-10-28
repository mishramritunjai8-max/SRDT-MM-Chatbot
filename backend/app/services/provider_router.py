import logging
from typing import List, Dict, Any, Optional, Tuple
from backend.app.core.config import settings
from backend.app.schemas.models import ModelInfo, AvailableModelsResponse
from backend.app.services.gemini_service import GeminiService
from backend.app.services.grok_service import GrokService
from backend.app.services.mock_service import MockService

logger = logging.getLogger(__name__)


class ProviderRouter:
    """
    Intelligent Model Router & Failover Engine.
    Coordinates Google Gemini and xAI Grok with automated switching and dynamic availability detection.
    """

    def __init__(self):
        self.gemini = GeminiService()
        self.grok = GrokService()
        self.mock = MockService()

    def get_available_models(self) -> AvailableModelsResponse:
        """Returns the list of configured and operational models."""
        models: List[ModelInfo] = []
        active_providers: List[str] = []

        # Grok models
        grok_ready = self.grok.is_configured()
        if grok_ready:
            active_providers.append("grok")
        models.append(ModelInfo(
            id="grok-2",
            name="Grok-2 (xAI)",
            provider="grok",
            is_available=grok_ready,
            supports_vision=False,
            supports_audio=False,
            description="High-reasoning and nuanced intelligence from xAI."
        ))
        models.append(ModelInfo(
            id="grok-2-vision-1212",
            name="Grok-2 Vision (xAI)",
            provider="grok",
            is_available=grok_ready,
            supports_vision=True,
            supports_audio=False,
            description="Multimodal reasoning over text and high-res images from xAI."
        ))

        # Gemini models
        gemini_ready = self.gemini.is_configured()
        if gemini_ready:
            active_providers.append("gemini")
        models.append(ModelInfo(
            id="gemini-3.5-flash",
            name="Gemini 3.5 Flash (Google)",
            provider="gemini",
            is_available=gemini_ready,
            supports_vision=True,
            supports_audio=True,
            description="Ultra-fast, natively multimodal model from Google."
        ))
        models.append(ModelInfo(
            id="gemini-1.5-flash",
            name="Gemini 1.5 Flash (Google)",
            provider="gemini",
            is_available=gemini_ready,
            supports_vision=True,
            supports_audio=True,
            description="Reliable lightweight multimodal model from Google."
        ))

        # Groq (100% Free Cloud LPU at https://console.groq.com/)
        groq_ready = bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY.strip()) > 5)
        if groq_ready:
            active_providers.append("groq")
        models.append(ModelInfo(
            id="llama-3.3-70b-versatile",
            name="Llama 3.3 70B (Groq Free)",
            provider="groq",
            is_available=groq_ready,
            supports_vision=False,
            supports_audio=False,
            description="100% Free ultra-fast 300+ tok/s high intelligence model from Groq Cloud."
        ))

        # Offline / Mock Fallback
        models.append(ModelInfo(
            id="mock",
            name="Offline Mock Provider",
            provider="mock",
            is_available=True,
            supports_vision=True,
            supports_audio=False,
            description="Deterministic local fallback provider for testing and offline development."
        ))

        default_model = "auto"
        if not grok_ready and not gemini_ready:
            default_model = "mock"

        return AvailableModelsResponse(
            models=models,
            default_model=default_model,
            auto_fallback_enabled=True,
            active_providers=active_providers
        )

    def generate_chat_response(
        self,
        prompt: str,
        preferred_model: Optional[str] = "auto",
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[str]] = None,
        emotional_mode: bool = False
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Generates chat response with auto-switching failover.
        Returns: (response_text, model_used, switched_from, switch_reason)
        """
        gemini_ready = self.gemini.is_configured()
        grok_ready = self.grok.is_configured()

        groq_ready = bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY.strip()) > 5)

        # If no keys configured, gracefully route to Mock
        if not gemini_ready and not grok_ready and not groq_ready:
            resp = self.mock.generate_chat_response(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                context_chunks=context_chunks,
                emotional_mode=emotional_mode
            )
            return resp, "mock", None, None

        # Determine attempt sequence based on preference
        model_choice = preferred_model or "auto"
        attempt_order: List[Tuple[str, str]] = []  # (provider, model_name)

        if model_choice == "auto":
            if gemini_ready:
                attempt_order.append(("gemini", settings.GEMINI_MODEL))
            if groq_ready:
                attempt_order.append(("groq", settings.GROQ_MODEL))
            if grok_ready:
                attempt_order.append(("grok", settings.GROK_MODEL))
        elif "groq" in model_choice.lower() or "llama" in model_choice.lower():
            if groq_ready:
                attempt_order.append(("groq", model_choice))
            if gemini_ready:
                attempt_order.append(("gemini", settings.GEMINI_MODEL))
            if grok_ready:
                attempt_order.append(("grok", settings.GROK_MODEL))
        elif "grok" in model_choice.lower():
            if grok_ready:
                attempt_order.append(("grok", model_choice))
            if groq_ready:
                attempt_order.append(("groq", settings.GROQ_MODEL))
            if gemini_ready:
                attempt_order.append(("gemini", settings.GEMINI_MODEL))
        elif "gemini" in model_choice.lower():
            if gemini_ready:
                attempt_order.append(("gemini", model_choice))
            if groq_ready:
                attempt_order.append(("groq", settings.GROQ_MODEL))
            if grok_ready:
                attempt_order.append(("grok", settings.GROK_MODEL))
        else:
            attempt_order.append(("mock", "mock"))

        # Always append mock at the very end of fallback chain
        attempt_order.append(("mock", "mock"))

        last_error = None
        first_attempt_model = attempt_order[0][1]

        for i, (provider, model_name) in enumerate(attempt_order):
            try:
                if provider == "groq":
                    resp = self._call_groq(
                        prompt=prompt,
                        model=model_name,
                        system_prompt=system_prompt,
                        history=history,
                        context_chunks=context_chunks
                    )
                elif provider == "grok":
                    resp = self.grok.generate_chat_response(
                        prompt=prompt,
                        model=model_name,
                        system_prompt=system_prompt,
                        history=history,
                        context_chunks=context_chunks
                    )
                elif provider == "gemini":
                    resp = self.gemini.generate_chat_response(
                        prompt=prompt,
                        model=model_name,
                        system_prompt=system_prompt,
                        history=history,
                        context_chunks=context_chunks
                    )
                else:  # mock
                    resp = self.mock.generate_chat_response(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history=history,
                        context_chunks=context_chunks,
                        emotional_mode=emotional_mode
                    )

                # If this wasn't the first attempt, auto-switching occurred!
                switched_from = first_attempt_model if i > 0 else None
                switch_reason = str(last_error) if i > 0 else None

                return resp, model_name, switched_from, switch_reason

            except Exception as e:
                logger.warning(f"Model attempt failed for {provider}/{model_name}: {e}")
                last_error = f"{type(e).__name__}: {str(e)}"
                continue

        # If everything failed, raise exception
        raise RuntimeError(f"All model providers failed. Last error: {last_error}")

    def analyze_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        preferred_model: Optional[str] = "auto"
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Multimodal image analysis with auto-switching failover.
        """
        gemini_ready = self.gemini.is_configured()
        grok_ready = self.grok.is_configured()

        if not gemini_ready and not grok_ready:
            resp = self.mock.analyze_image(prompt, image_bytes, mime_type)
            return resp, "mock", None, None

        attempt_order: List[Tuple[str, str]] = []
        if preferred_model == "auto" or "gemini" in (preferred_model or ""):
            if gemini_ready:
                attempt_order.append(("gemini", settings.GEMINI_MODEL))
            if grok_ready:
                attempt_order.append(("grok", settings.GROK_VISION_MODEL))
        else:
            if grok_ready:
                attempt_order.append(("grok", settings.GROK_VISION_MODEL))
            if gemini_ready:
                attempt_order.append(("gemini", settings.GEMINI_MODEL))

        attempt_order.append(("mock", "mock"))
        first_model = attempt_order[0][1]
        last_error = None

        for i, (provider, model_name) in enumerate(attempt_order):
            try:
                if provider == "gemini":
                    resp = self.gemini.analyze_image(prompt, image_bytes, mime_type, model=model_name)
                elif provider == "grok":
                    resp = self.grok.analyze_image(prompt, image_bytes, mime_type, model=model_name)
                else:
                    resp = self.mock.analyze_image(prompt, image_bytes, mime_type)

                switched_from = first_model if i > 0 else None
                switch_reason = str(last_error) if i > 0 else None
                return resp, model_name, switched_from, switch_reason
            except Exception as e:
                logger.warning(f"Vision attempt failed for {provider}/{model_name}: {e}")
                last_error = str(e)
                continue

        raise RuntimeError(f"All vision providers failed: {last_error}")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini if configured, else deterministic fallback."""
        if self.gemini.is_configured():
            try:
                return self.gemini.get_embedding(text)
            except Exception as e:
                logger.warning(f"Gemini embedding failed, falling back to local: {e}")
        return self.mock.get_embedding(text)

    def _call_groq(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[str]] = None
    ) -> str:
        """Call Groq Cloud free endpoint."""
        import httpx
        selected_model = model or settings.GROQ_MODEL
        endpoint = f"{settings.GROQ_BASE_URL.rstrip('/')}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful and intelligent AI assistant."})

        if history:
            for msg in history:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        final_prompt = prompt
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            final_prompt = (
                f"Use ONLY the following context to answer:\n\n{context_text}\n\nQuestion: {prompt}"
            )

        messages.append({"role": "user", "content": final_prompt})

        payload = {"model": selected_model, "messages": messages, "temperature": 0.7}
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            if resp.status_code == 429:
                raise RuntimeError("Groq rate limit exceeded (HTTP 429).")
            if resp.status_code != 200:
                raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"]
