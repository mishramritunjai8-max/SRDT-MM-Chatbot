import io
from PIL import Image
from backend.app.services.image_upscaler_service import PromptOptimizer, ImageUpscalerService


def test_prompt_optimizer_enhancement():
    raw_prompt = "a laughing cat"
    optimized = PromptOptimizer.optimize(raw_prompt, style="flux-realism")
    assert "a laughing cat" in optimized
    assert "8k" in optimized
    assert "sharp focus" in optimized


def test_prompt_optimizer_already_detailed():
    detailed_prompt = "8k resolution masterpiece of a samurai"
    optimized = PromptOptimizer.optimize(detailed_prompt, style="flux")
    # Should not duplicate 8k
    assert optimized.lower().count("8k") == 1


def test_image_upscaler_service_2x():
    img = Image.new("RGB", (300, 200), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    
    enhanced_bytes, orig_size, new_size = ImageUpscalerService.upscale_and_enhance(
        image_bytes=buf.getvalue(),
        scale_factor=2.0,
        clarity_boost=True
    )
    
    assert orig_size == (300, 200)
    assert new_size == (600, 400)
    assert len(enhanced_bytes) > 0
    
    # Check that output is valid readable image
    out_img = Image.open(io.BytesIO(enhanced_bytes))
    assert out_img.size == (600, 400)


def test_image_upscaler_service_target_dimensions():
    img = Image.new("RGB", (512, 512), color=(80, 120, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    
    enhanced_bytes, orig_size, new_size = ImageUpscalerService.upscale_and_enhance(
        image_bytes=buf.getvalue(),
        target_width=1920,
        target_height=1080,
        clarity_boost=True
    )
    
    assert orig_size == (512, 512)
    assert new_size == (1920, 1080)
    out_img = Image.open(io.BytesIO(enhanced_bytes))
    assert out_img.size == (1920, 1080)
