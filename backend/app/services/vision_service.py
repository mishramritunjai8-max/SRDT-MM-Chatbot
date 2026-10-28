import logging
from typing import Tuple, Optional
from backend.app.services.provider_router import ProviderRouter

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(self):
        self.router = ProviderRouter()

    def analyze_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        preferred_model: Optional[str] = "auto"
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Processes image with either Gemini or Grok Vision, auto-switching if limits are reached.
        """
        return self.router.analyze_image(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            preferred_model=preferred_model
        )
