def test_get_available_models(client):
    response = client.get("/models/available")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 3
    
    model_ids = [m["id"] for m in data["models"]]
    assert "grok-2" in model_ids
    assert "gemini-3.5-flash" in model_ids
    assert "mock" in model_ids
    assert data["auto_fallback_enabled"] is True
