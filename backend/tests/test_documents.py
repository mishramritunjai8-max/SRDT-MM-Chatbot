import io


def test_document_upload_and_list(client):
    file_content = b"Artificial Intelligence is a branch of computer science that builds smart machines."
    files = {"file": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}
    
    upload_resp = client.post("/documents/upload", files=files)
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    assert doc_data["filename"] == "test_doc.txt"
    assert doc_data["status"] == "processed"
    assert doc_data["chunk_count"] >= 1
    doc_id = doc_data["id"]

    # List documents
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert any(d["id"] == doc_id for d in docs)

    # Get single document
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == 200
    single_doc = get_resp.json()
    assert single_doc["id"] == doc_id
    assert "Artificial Intelligence" in single_doc["original_text"]

    # Delete document
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200

    # Verify deleted
    get_del = client.get(f"/documents/{doc_id}")
    assert get_del.status_code == 404


def test_generated_document_versioning(client):
    create_payload = {
        "title": "Project Summary",
        "content": "Initial report on the multimodal chatbot architecture.",
        "format": "markdown"
    }
    create_resp = client.post("/documents/create", json=create_payload)
    assert create_resp.status_code == 201
    gen_doc = create_resp.json()
    assert gen_doc["title"] == "Project Summary"
    assert gen_doc["version"] == 1
    gen_doc_id = gen_doc["id"]

    # Update version
    update_payload = {
        "content": "Updated report with Grok and Gemini dual-provider details.",
        "revision_notes": "Added multi-model fallback"
    }
    update_resp = client.put(f"/documents/generated/{gen_doc_id}", json=update_payload)
    assert update_resp.status_code == 200
    updated_doc = update_resp.json()
    assert updated_doc["version"] == 2
    assert "Grok and Gemini" in updated_doc["content"]
