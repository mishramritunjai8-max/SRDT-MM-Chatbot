def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["database"]["status"] == "healthy"
    assert "providers" in data
    assert data["providers"]["mock"] == "always_ready"
    assert data["auto_fallback_enabled"] is True
