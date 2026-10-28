import io


def test_rag_chunking_and_grounded_answer(client):
    # 1. Upload a document with specific factual content
    doc_text = (
        "Quantum superposition is a fundamental principle of quantum mechanics. "
        "It states that much like waves in classical physics, any two or more quantum states "
        "can be added together ('superposed') and the result will be another valid quantum state. "
        "The project code name is Project Neptune."
    )
    files = {"file": ("quantum_notes.txt", io.BytesIO(doc_text.encode("utf-8")), "text/plain")}
    upload_resp = client.post("/documents/upload", files=files)
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["id"]

    # 2. Query with RAG enabled
    chat_payload = {
        "message": "What is the secret project code name?",
        "use_rag": True,
        "document_id": doc_id,
        "model": "mock"
    }
    chat_resp = client.post("/chat", json=chat_payload)
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert "response" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["title"] == "quantum_notes.txt"

    # Cleanup
    client.delete(f"/documents/{doc_id}")
