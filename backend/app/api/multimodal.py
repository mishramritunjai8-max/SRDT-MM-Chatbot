from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from typing import Optional
from backend.app.services.vision_service import VisionService
from backend.app.services.audio_service import AudioService
from backend.app.services.image_generation_service import ImageGenerationService

router = APIRouter(prefix="/multimodal", tags=["Multimodal (Vision & Audio)"])
vision_service = VisionService()
audio_service = AudioService()
image_gen_service = ImageGenerationService()


@router.post("/vision")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("Describe and analyze what you see in this image in detail."),
    model: Optional[str] = Form("auto")
):
    """
    Multimodal image comprehension supporting JPG, JPEG, and PNG.
    Routes to Gemini or Grok Vision with auto-failover if limits are hit.
    """
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. Supported: JPG, JPEG, PNG, WEBP"
        )

    try:
        image_bytes = await file.read()
        resp_text, model_used, switched_from, switch_reason = vision_service.analyze_image(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=file.content_type,
            preferred_model=model
        )
        return {
            "response": resp_text,
            "model_used": model_used,
            "switched_from": switched_from,
            "switch_reason": switch_reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize")
def text_to_speech(text: str = Form(...)):
    """Convert text response to spoken MP3 audio."""
    try:
        audio_bytes = audio_service.synthesize_speech(text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def speech_to_text(file: UploadFile = File(...)):
    """Transcribe speech audio file to text."""
    try:
        audio_bytes = await file.read()
        transcription = audio_service.transcribe_audio(audio_bytes, file.filename or "audio.wav")
        return {"transcript": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-image")
def generate_image_endpoint(
    prompt: str = Form(...),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(1024),
    model: Optional[str] = Form("flux"),
    upscale: Optional[bool] = Form(True),
    scale_factor: Optional[float] = Form(2.0),
    enhance_prompt: Optional[bool] = Form(True)
):
    """
    Text-to-Image generation endpoint.
    Synthesizes high-resolution, sharp, crystal-clear images using FLUX / Stable Diffusion
    with AI prompt optimization and super-resolution upscaling.
    """
    try:
        model_str = model if isinstance(model, str) else "flux"
        width_int = int(width) if isinstance(width, (int, str)) and str(width).isdigit() else 1024
        height_int = int(height) if isinstance(height, (int, str)) and str(height).isdigit() else 1024
        upscale_bool = upscale if isinstance(upscale, bool) else (str(upscale).lower() in ["true", "1", "yes"] if upscale is not None else True)
        scale_factor_float = float(scale_factor) if isinstance(scale_factor, (int, float, str)) and str(scale_factor).replace(".", "", 1).isdigit() else 2.0
        enhance_prompt_bool = enhance_prompt if isinstance(enhance_prompt, bool) else (str(enhance_prompt).lower() in ["true", "1", "yes"] if enhance_prompt is not None else True)

        image_bytes, model_used = image_gen_service.generate_image(
            prompt=prompt,
            width=width_int,
            height=height_int,
            model=model_str,
            upscale=upscale_bool,
            scale_factor=scale_factor_float,
            enhance_prompt=enhance_prompt_bool
        )
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={"x-model-used": model_used}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upscale-image")
async def upscale_image_endpoint(
    file: UploadFile = File(...),
    scale_factor: Optional[float] = Form(2.0),
    clarity_boost: Optional[bool] = Form(True)
):
    """
    AI Super-Resolution and Clarity Upscaling endpoint.
    Upscales any image (2x / 4x) using sub-pixel reconstruction, adaptive unsharp masking,
    and artifact suppression without quality loss.
    """
    try:
        image_bytes = await file.read()
        enhanced_bytes, orig_size, new_size = image_gen_service.upscale_image(
            image_bytes=image_bytes,
            scale_factor=scale_factor or 2.0,
            clarity_boost=True if clarity_boost is None else clarity_boost
        )
        return Response(
            content=enhanced_bytes,
            media_type="image/jpeg",
            headers={
                "x-original-size": f"{orig_size[0]}x{orig_size[1]}",
                "x-upscaled-size": f"{new_size[0]}x{new_size[1]}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
