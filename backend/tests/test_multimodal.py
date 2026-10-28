import io


def test_vision_image_analysis(client):
    # Create a small valid 1x1 GIF/PNG dummy byte stream
    dummy_image = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    files = {"file": ("test_image.png", io.BytesIO(dummy_image), "image/png")}
    data = {"prompt": "What color is this pixel?", "model": "mock"}

    response = client.post("/multimodal/vision", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert "response" in res
    assert "model_used" in res


def test_text_to_speech_synthesis(client):
    data = {"text": "Hello world, this is a test of speech synthesis."}
    response = client.post("/multimodal/synthesize", data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 0


def test_text_to_image_generation(client):
    data = {"prompt": "A serene mountain lake at sunrise with crystal reflections", "model": "flux", "upscale": "false"}
    response = client.post("/multimodal/generate-image", data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000


def test_text_to_image_generation_with_upscaling(client):
    data = {
        "prompt": "Cyberpunk city with neon lights and flying cars",
        "model": "flux-realism",
        "upscale": "true",
        "scale_factor": "2.0",
        "enhance_prompt": "true"
    }
    response = client.post("/multimodal/generate-image", data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "ai-upscale" in response.headers.get("x-model-used", "")
    assert len(response.content) > 1000


def test_image_upscaling_endpoint(client):
    from PIL import Image
    dummy_img = Image.new("RGB", (256, 256), color=(50, 100, 150))
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    
    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    data = {"scale_factor": "2.0", "clarity_boost": "true"}
    
    response = client.post("/multimodal/upscale-image", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers.get("x-original-size") == "256x256"
    assert response.headers.get("x-upscaled-size") == "512x512"
    assert len(response.content) > 500


