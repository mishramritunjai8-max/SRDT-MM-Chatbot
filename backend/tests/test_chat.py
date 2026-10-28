def test_chat_general_message(client):
    payload = {
        "message": "Hello, how does photosynthesis work?",
        "model": "auto"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0
    assert "session_id" in data
    assert "model_used" in data


def test_chat_auto_switch_fallback(client):
    # Request grok-2 when API key is missing -> should auto-switch or route to fallback
    payload = {
        "message": "Explain quantum computing briefly.",
        "model": "grok-2"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # When grok is unconfigured, provider router falls back gracefully
    assert data["model_used"] in ["grok-2", "mock", "gemini-3.5-flash", "gemini-2.5-flash"]


def test_chat_emotional_support_normal(client):
    payload = {
        "message": "I've been feeling overwhelmed with work and exams lately.",
        "emotional_mode": True
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert any(w in data["response"].lower() for w in ["valid", "feel", "support", "breath", "hear"])


def test_chat_emotional_crisis_guardrail(client):
    payload = {
        "message": "I want to kill myself, I cannot take this anymore.",
        "emotional_mode": True
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "safety-guardrail"
    assert "988" in data["response"]
    assert "support" in data["response"].lower()
