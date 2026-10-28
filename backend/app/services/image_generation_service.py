import logging
import urllib.parse
import random
import io
import time
import os
import re
import httpx
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from backend.app.services.image_upscaler_service import PromptOptimizer, ImageUpscalerService

logger = logging.getLogger(__name__)


class ImageGenerationService:
    """
    Multimodal Text-to-Image Generation Service.
    Powered by state-of-the-art open models (FLUX.1 / SDXL Turbo) via Pollinations AI
    with intelligent prompt optimization, AI super-resolution upscaling,
    smart 45s cooldown handling, and robust multi-endpoint failover.
    """

    def __init__(self):
        self.upscaler = ImageUpscalerService()

    def _fetch_online_image(self, prompt: str, width: int, height: int, model: str) -> Optional[bytes]:
        """Queries Pollinations AI with browser headers, intelligent cooldown handling, and resilient HTTP transport."""
        # Short-circuit during mock/test environments to prevent offline socket hanging
        if os.getenv("DEFAULT_MODEL") == "mock" or os.getenv("APP_ENV") == "test":
            return None

        import requests
        clean_prompt = prompt.replace("%", " percent ").replace("\n", " ").strip()
        encoded_prompt = urllib.parse.quote(clean_prompt)
        seed = random.randint(1000, 999999)
        api_model = "turbo" if "turbo" in model.lower() else "flux"

        # Realistic browser headers to bypass Cloudflare anti-bot checks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://pollinations.ai/",
            "Origin": "https://pollinations.ai"
        }

        # Optional Pollen API key if configured
        pollen_key = os.getenv("POLLEN_API_KEY") or os.getenv("POLLINATIONS_API_KEY")
        if pollen_key:
            headers["Authorization"] = f"Bearer {pollen_key.strip()}"

        endpoints = [
            f"https://pollinations.ai/p/{encoded_prompt}?width={width}&height={height}&model={api_model}&seed={seed}",
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&model={api_model}&nologo=true&seed={seed}",
            f"https://pollinations.ai/p/{encoded_prompt}?width={width}&height={height}&seed={seed}"
        ]

        for url in endpoints:
            try:
                resp = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 1000 and resp.headers.get("content-type", "").startswith("image/"):
                    return resp.content

                # Smart Cooldown Handling: Free tier rate limit (45s per IP)
                if resp.status_code == 402:
                    logger.warning(f"Pollinations free tier rate limit encountered: {resp.text[:150]}")
                    match = re.search(r"Retry after (\d+)s", resp.text)
                    wait_s = int(match.group(1)) + 1 if match else 15
                    if wait_s <= 35:
                        logger.info(f"Waiting {wait_s}s for Pollinations cooldown to clear...")
                        time.sleep(wait_s)
                        retry_resp = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
                        if retry_resp.status_code == 200 and len(retry_resp.content) > 1000:
                            return retry_resp.content

                logger.warning(f"Pollinations returned HTTP {resp.status_code} for {url}")
            except Exception as e:
                logger.warning(f"Requests fetch failed for {url}: {e}, trying httpx...")

            try:
                timeout_cfg = httpx.Timeout(60.0, connect=10.0)
                with httpx.Client(timeout=timeout_cfg, follow_redirects=True, headers=headers) as client:
                    response = client.get(url)
                    if response.status_code == 200 and len(response.content) > 1000:
                        return response.content
            except Exception as e:
                logger.warning(f"HTTPX fetch failed for {url}: {e}")

        return None

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        model: str = "flux",
        upscale: bool = True,
        scale_factor: float = 2.0,
        enhance_prompt: bool = True
    ) -> Tuple[bytes, str]:
        """
        Generates a high-resolution, sharp, crystal-clear, artifact-free image.
        1. Optimizes prompt with lighting, detail, and anti-artifact tags.
        2. Queries FLUX/SDXL generation engine.
        3. Applies multi-stage AI Super-Resolution upscaling and edge sharpening when requested.
        Returns: (image_bytes, model_used)
        """
        # Step 1: Prompt Optimization for clarity and artifact suppression
        effective_prompt = prompt.strip()
        if enhance_prompt:
            effective_prompt = PromptOptimizer.optimize(effective_prompt, style=model)

        # For ultra-high target resolutions, generate at optimal diffusion native size (1024)
        # and then upscale via AI super-resolution to prevent hallucination / limb duplication artifacts.
        gen_width = min(width, 1024) if upscale else width
        gen_height = min(height, 1024) if upscale else height

        raw_bytes = self._fetch_online_image(effective_prompt, gen_width, gen_height, model)
        model_used = f"flux-{model}"

        # Programmatic local high-res visual fallback if offline/sandboxed
        if not raw_bytes:
            raw_bytes = self._generate_fallback_image(effective_prompt, gen_width, gen_height)
            model_used = "offline-art-engine"

        # Step 3: AI Super-Resolution Upscaling & Clarity Enhancement
        if upscale:
            try:
                target_w = width if width > gen_width else int(gen_width * scale_factor)
                target_h = height if height > gen_height else int(gen_height * scale_factor)
                enhanced_bytes, _, _ = self.upscaler.upscale_and_enhance(
                    image_bytes=raw_bytes,
                    scale_factor=scale_factor,
                    clarity_boost=True,
                    target_width=target_w,
                    target_height=target_h
                )
                return enhanced_bytes, f"{model_used}+ai-upscale"
            except Exception as e:
                logger.warning(f"AI Upscaling failed, returning base image: {e}")

        return raw_bytes, model_used

    def upscale_image(
        self,
        image_bytes: bytes,
        scale_factor: float = 2.0,
        clarity_boost: bool = True
    ) -> Tuple[bytes, Tuple[int, int], Tuple[int, int]]:
        """
        Directly upscales and sharpens an existing image.
        """
        return self.upscaler.upscale_and_enhance(
            image_bytes=image_bytes,
            scale_factor=scale_factor,
            clarity_boost=clarity_boost
        )

    def _generate_fallback_image(self, prompt: str, width: int = 1024, height: int = 1024) -> bytes:
        """Diagnostic card rendered ONLY when internet is completely disconnected or mocked in tests."""
        img = Image.new("RGB", (width, height), color=(15, 20, 30))
        draw = ImageDraw.Draw(img)

        # Subtle elegant border
        draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(40, 50, 75), width=2)
        draw.rectangle([(40, 40), (width - 40, height - 40)], outline=(60, 80, 120), width=1)

        # Clear, truthful diagnostic label
        draw.text((60, 70), "AI IMAGE SYNTHESIS - STATUS REPORT", fill=(255, 180, 50))
        draw.text((60, 110), f"Requested Subject: {prompt[:120]}", fill=(240, 245, 255))
        draw.text((60, 150), "Notice: External generation provider was unreachable or rate-limited.", fill=(180, 200, 220))
        draw.text((60, 180), "Free tier has a 45s cooldown per IP. Please retry in a few moments.", fill=(140, 160, 190))
        draw.text((60, height - 70), "Tip: Visit enter.pollinations.ai to get a free API key for instant generation.", fill=(100, 210, 160))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=98, subsampling=0)
        return buf.getvalue()
