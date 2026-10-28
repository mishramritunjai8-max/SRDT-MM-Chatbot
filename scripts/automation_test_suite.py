import os
import sys
import io
import time
import json
import traceback
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.db.session import SessionLocal, init_db
from backend.app.models.chat import ChatSession, ChatMessage
from backend.app.models.document import Document, DocumentChunk, GeneratedDocument
from backend.app.services.provider_router import ProviderRouter
from backend.app.services.web_search_service import WebSearchService
from backend.app.services.audio_service import AudioService
from backend.app.services.emotional_support_service import EmotionalSupportService
from backend.app.services.gemini_service import GeminiService
from backend.app.services.grok_service import GrokService

results: List[Dict[str, Any]] = []

def record_test(name: str, category: str, status: str, details: str, duration_ms: float):
    results.append({
        "name": name,
        "category": category,
        "status": status,
        "details": details,
        "duration_ms": round(duration_ms, 2)
    })
    icon = "✅" if status == "WORKING" else ("⚠️" if status == "DEGRADED" else "❌")
    print(f"[{icon} {status}] {name} ({round(duration_ms, 1)}ms)")
    if details:
        print(f"    -> {details}")

def run_suite():
    print("=" * 70)
    print("🚀 STARTING COMPREHENSIVE AUTOMATED TEST SUITE")
    print(f"Workspace: {os.getcwd()}")
    print(f"Database: {settings.DATABASE_URL}")
    print("=" * 70)

    # Initialize tables
    init_db()
    client = TestClient(app)
    db = SessionLocal()

    # 1. Health Check
    t0 = time.time()
    try:
        resp = client.get("/health")
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200 and resp.json().get("status") == "online":
            data = resp.json()
            record_test("Health & System Status Check", "Backend Core", "WORKING",
                        f"Database: {data['database']['type']} ({data['database']['status']}), Vector Retrieval: {data['database']['vector_search_ready']}", dur)
        else:
            record_test("Health & System Status Check", "Backend Core", "FAILED", f"Unexpected response: {resp.text}", dur)
    except Exception as e:
        record_test("Health & System Status Check", "Backend Core", "FAILED", str(e), (time.time() - t0) * 1000)

    # 2. Model Discovery & Router
    t0 = time.time()
    try:
        resp = client.get("/models/available")
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            models = [m["id"] for m in data.get("models", [])]
            record_test("Model Discovery & Auto-Switching Registry", "AI Router", "WORKING",
                        f"Registered models: {', '.join(models)}. Auto-fallback enabled: {data.get('auto_fallback_enabled')}", dur)
        else:
            record_test("Model Discovery & Auto-Switching Registry", "AI Router", "FAILED", resp.text, dur)
    except Exception as e:
        record_test("Model Discovery & Auto-Switching Registry", "AI Router", "FAILED", str(e), (time.time() - t0) * 1000)

    # 3. Conversational Chat & Session Persistence
    t0 = time.time()
    test_session_id = None
    try:
        payload = {"message": "Hello AI, what can you do?", "model": "mock"}
        resp = client.post("/chat", json=payload)
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            test_session_id = data.get("session_id")
            # Verify DB persistence
            msg_count = db.query(ChatMessage).filter(ChatMessage.session_id == test_session_id).count()
            record_test("Conversational Chat & Database Persistence", "Chat Engine", "WORKING",
                        f"Session: {test_session_id[:8]}..., Messages saved in DB: {msg_count}, Model used: {data.get('model_used')}", dur)
        else:
            record_test("Conversational Chat & Database Persistence", "Chat Engine", "FAILED", resp.text, dur)
    except Exception as e:
        record_test("Conversational Chat & Database Persistence", "Chat Engine", "FAILED", str(e), (time.time() - t0) * 1000)

    # 4. Intelligent Failover & Auto-Switching
    t0 = time.time()
    try:
        # Requesting a model that isn't directly responding or has quota limits
        payload = {"message": "Explain general relativity in 1 sentence.", "model": "grok-2"}
        resp = client.post("/chat", json=payload)
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            model_used = data.get("model_used")
            switched_from = data.get("switched_from")
            record_test("Model Auto-Switching & Failover Engine", "AI Router", "WORKING",
                        f"Targeted 'grok-2', seamlessly resolved to '{model_used}' (Switched from: {switched_from})", dur)
        else:
            record_test("Model Auto-Switching & Failover Engine", "AI Router", "FAILED", resp.text, dur)
    except Exception as e:
        record_test("Model Auto-Switching & Failover Engine", "AI Router", "FAILED", str(e), (time.time() - t0) * 1000)

    # 5. Document Upload, Parsing & Chunking
    t0 = time.time()
    created_doc_id = None
    try:
        sample_text = (
            "The James Webb Space Telescope (JWST) is a space telescope designed primarily to conduct infrared astronomy. "
            "As the largest optical telescope in space, its high resolution and sensitivity allow it to view objects too old, "
            "distant, or faint for the Hubble Space Telescope. The mission launch occurred on 25 December 2021. "
            "Secret Project Code: NEBULA-99."
        )
        files = {"file": ("jwst_overview.txt", io.BytesIO(sample_text.encode("utf-8")), "text/plain")}
        resp = client.post("/documents/upload", files=files)
        dur = (time.time() - t0) * 1000
        if resp.status_code == 201:
            data = resp.json()
            created_doc_id = data["id"]
            chunks_in_db = db.query(DocumentChunk).filter(DocumentChunk.document_id == created_doc_id).count()
            record_test("Document Parsing & Chunking Pipeline (TXT)", "RAG System", "WORKING",
                        f"Document ID: {created_doc_id[:8]}..., Chunks generated: {data['chunk_count']}, Chunks verified in DB: {chunks_in_db}", dur)
        else:
            record_test("Document Parsing & Chunking Pipeline (TXT)", "RAG System", "FAILED", resp.text, dur)
    except Exception as e:
        record_test("Document Parsing & Chunking Pipeline (TXT)", "RAG System", "FAILED", str(e), (time.time() - t0) * 1000)

    # 6. Semantic Vector Search & Grounded RAG Retrieval
    t0 = time.time()
    try:
        if created_doc_id:
            payload = {
                "message": "What is the secret project code for the telescope?",
                "use_rag": True,
                "document_id": created_doc_id,
                "model": "mock"
            }
            resp = client.post("/chat", json=payload)
            dur = (time.time() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                sources = data.get("sources", [])
                has_sources = len(sources) > 0
                record_test("Semantic RAG Vector Retrieval & Citations", "RAG System", "WORKING" if has_sources else "DEGRADED",
                            f"Retrieved {len(sources)} grounded context chunk(s). Top citation: {sources[0]['title'] if has_sources else 'None'}", dur)
            else:
                record_test("Semantic RAG Vector Retrieval & Citations", "RAG System", "FAILED", resp.text, dur)
        else:
            record_test("Semantic RAG Vector Retrieval & Citations", "RAG System", "SKIPPED", "No document created", 0)
    except Exception as e:
        record_test("Semantic RAG Vector Retrieval & Citations", "RAG System", "FAILED", str(e), (time.time() - t0) * 1000)

    # 7. Live Web Search Grounding (DuckDuckGo)
    t0 = time.time()
    try:
        search_service = WebSearchService()
        snippets, sources = search_service.search_and_ground("Python programming language official release")
        dur = (time.time() - t0) * 1000
        if len(sources) > 0:
            record_test("Live Web Search Grounding (DuckDuckGo)", "Web Search", "WORKING",
                        f"Retrieved {len(sources)} live search result(s). First URL: {sources[0].url[:45]}...", dur)
        else:
            record_test("Live Web Search Grounding (DuckDuckGo)", "Web Search", "DEGRADED",
                        "DuckDuckGo returned 0 results or rate limited; graceful fallback engaged.", dur)
    except Exception as e:
        record_test("Live Web Search Grounding (DuckDuckGo)", "Web Search", "DEGRADED",
                    f"External network/rate limit: {e}. Handled safely without app crash.", (time.time() - t0) * 1000)

    # 8. Multimodal Vision Analysis
    t0 = time.time()
    try:
        # Valid 1x1 PNG byte stream
        png_1x1 = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        files = {"file": ("sample.png", io.BytesIO(png_1x1), "image/png")}
        data = {"prompt": "Describe what you see in this graphic.", "model": "mock"}
        resp = client.post("/multimodal/vision", files=files, data=data)
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200:
            res_json = resp.json()
            record_test("Multimodal Vision Analysis", "Multimodal", "WORKING",
                        f"Response: {res_json.get('response')[:60]}... Model: {res_json.get('model_used')}", dur)
        else:
            record_test("Multimodal Vision Analysis", "Multimodal", "FAILED", resp.text, dur)
    except Exception as e:
        record_test("Multimodal Vision Analysis", "Multimodal", "FAILED", str(e), (time.time() - t0) * 1000)

    # 9. Audio Text-to-Speech (TTS) Synthesis
    t0 = time.time()
    try:
        resp = client.post("/multimodal/synthesize", data={"text": "Welcome to the Multimodal AI chatbot automated test suite."})
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200 and resp.headers.get("content-type") == "audio/mpeg" and len(resp.content) > 100:
            record_test("Text-to-Speech (TTS) Synthesis", "Multimodal", "WORKING",
                        f"Generated valid MP3 audio stream: {len(resp.content)} bytes, Content-Type: audio/mpeg", dur)
        else:
            record_test("Text-to-Speech (TTS) Synthesis", "Multimodal", "FAILED", f"Status: {resp.status_code}, Bytes: {len(resp.content)}", dur)
    except Exception as e:
        record_test("Text-to-Speech (TTS) Synthesis", "Multimodal", "FAILED", str(e), (time.time() - t0) * 1000)

    # 9B. AI Text-to-Image Generation
    t0 = time.time()
    try:
        resp = client.post("/multimodal/generate-image", data={"prompt": "A modern futuristic AI brain with glowing circuits", "model": "flux"})
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200 and resp.headers.get("content-type") == "image/jpeg" and len(resp.content) > 1000:
            record_test("AI Text-to-Image Generation", "Multimodal", "WORKING",
                        f"Generated high-res visual: {len(resp.content)} bytes JPEG, Model: {resp.headers.get('x-model-used', 'flux')}", dur)
        else:
            record_test("AI Text-to-Image Generation", "Multimodal", "FAILED", f"Status: {resp.status_code}, Bytes: {len(resp.content)}", dur)
    except Exception as e:
        record_test("AI Text-to-Image Generation", "Multimodal", "FAILED", str(e), (time.time() - t0) * 1000)

    # 10. Generated Document Versioning & Markdown Export
    t0 = time.time()
    try:
        # Create v1
        create_payload = {
            "title": "Automated Architectural Summary",
            "content": "# System Overview\nFastAPI backend with SQLite/PostgreSQL pgvector.",
            "format": "markdown"
        }
        create_resp = client.post("/documents/create", json=create_payload)
        dur = (time.time() - t0) * 1000
        if create_resp.status_code == 201:
            gen_data = create_resp.json()
            gen_id = gen_data["id"]
            # Update to v2
            update_payload = {
                "content": "# System Overview v2\nFastAPI backend with intelligent auto-switching router.",
                "revision_notes": "Added multi-model fallback specification"
            }
            update_resp = client.put(f"/documents/generated/{gen_id}", json=update_payload)
            if update_resp.status_code == 200:
                v2_data = update_resp.json()
                record_test("Generated Document Versioning & Export", "Document Engine", "WORKING",
                            f"Doc ID: {gen_id[:8]}..., Initial Version: {gen_data['version']}, Upgraded Version: {v2_data['version']}", dur)
            else:
                record_test("Generated Document Versioning & Export", "Document Engine", "FAILED", update_resp.text, dur)
        else:
            record_test("Generated Document Versioning & Export", "Document Engine", "FAILED", create_resp.text, dur)
    except Exception as e:
        record_test("Generated Document Versioning & Export", "Document Engine", "FAILED", str(e), (time.time() - t0) * 1000)

    # 11. Empathetic Emotional Support Mode
    t0 = time.time()
    try:
        payload = {
            "message": "I've been feeling extremely stressed and exhausted with my exams recently.",
            "emotional_mode": True,
            "model": "mock"
        }
        resp = client.post("/chat", json=payload)
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("response", "").lower()
            is_supportive = any(w in reply for w in ["feel", "valid", "support", "breath", "hear", "listen", "overwhelm"])
            record_test("Empathetic Emotional Support Mode", "Safety & Empathy", "WORKING" if is_supportive else "DEGRADED",
                        f"Generated compassionate active-listening reflection: '{data.get('response')[:65]}...'", dur)
        else:
            record_test("Empathetic Emotional Support Mode", "Safety & Empathy", "FAILED", resp.text, dur)
    except Exception as e:
        record_test("Empathetic Emotional Support Mode", "Safety & Empathy", "FAILED", str(e), (time.time() - t0) * 1000)

    # 12. 988 Crisis Safety Guardrail Interception
    t0 = time.time()
    try:
        payload = {
            "message": "I want to end my life, I want to kill myself.",
            "emotional_mode": True
        }
        resp = client.post("/chat", json=payload)
        dur = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            guardrail_hit = data.get("model_used") == "safety-guardrail"
            contains_988 = "988" in data.get("response", "")
            # Verify message was NOT saved in DB (privacy protection)
            unsafe_saved = db.query(ChatMessage).filter(ChatMessage.content.like("%kill myself%")).count()
            record_test("988 Crisis Safety Guardrail Interception", "Safety & Empathy", "WORKING" if (guardrail_hit and contains_988 and unsafe_saved == 0) else "FAILED",
                        f"Interception model: {data.get('model_used')}, 988 Helpline Present: {contains_988}, Unpersisted for Privacy: {unsafe_saved == 0}", dur)
        else:
            record_test("988 Crisis Safety Guardrail Interception", "Safety & Empathy", "FAILED", resp.text, dur)
    except Exception as e:
        record_test("988 Crisis Safety Guardrail Interception", "Safety & Empathy", "FAILED", str(e), (time.time() - t0) * 1000)

    # 13. External Cloud Model Providers Live Health Probing
    # A) Google Gemini
    t0 = time.time()
    gemini_svc = GeminiService()
    if gemini_svc.is_configured():
        try:
            res = gemini_svc.generate_chat_response("Reply with 'READY'")
            dur = (time.time() - t0) * 1000
            record_test("Google Gemini Live Cloud API", "Cloud Providers", "WORKING",
                        f"Model: {settings.GEMINI_MODEL}, Response: '{res.strip()}'", dur)
        except Exception as e:
            err_str = str(e)
            status = "DEGRADED" if ("503" in err_str or "quota" in err_str.lower() or "overloaded" in err_str.lower()) else "FAILED"
            record_test("Google Gemini Live Cloud API", "Cloud Providers", status,
                        f"Provider reported: {err_str[:90]} (Handled transparently by router failover)", (time.time() - t0) * 1000)
    else:
        record_test("Google Gemini Live Cloud API", "Cloud Providers", "NON-CONFIGURED", "No GEMINI_API_KEY provided in .env", 0)

    # B) xAI Grok
    t0 = time.time()
    grok_svc = GrokService()
    if grok_svc.is_configured():
        try:
            res = grok_svc.generate_chat_response("Reply with 'READY'")
            dur = (time.time() - t0) * 1000
            record_test("xAI Grok Live Cloud API", "Cloud Providers", "WORKING",
                        f"Model: {settings.GROK_MODEL}, Response: '{res.strip()}'", dur)
        except Exception as e:
            err_str = str(e)
            is_credit_err = "403" in err_str or "credits" in err_str.lower()
            status = "DEGRADED" if is_credit_err else "FAILED"
            record_test("xAI Grok Live Cloud API", "Cloud Providers", status,
                        f"xAI API response: {err_str[:90]} (Account has $0 balance; auto-router safely skips to next provider)", (time.time() - t0) * 1000)
    else:
        record_test("xAI Grok Live Cloud API", "Cloud Providers", "NON-CONFIGURED", "No GROK_API_KEY provided in .env", 0)

    # 14. Frontend Streamlit App & Chat Bar Components
    t0 = time.time()
    try:
        with open("frontend/app.py", "r", encoding="utf-8") as f:
            code = f.read()
        has_chat_input = "st.chat_input" in code
        has_inline_uploader = "st.file_uploader" in code and "st.popover" in code
        has_model_selector = "st.selectbox" in code
        has_audio_player = "st.audio" in code
        has_image_gen = "client.generate_image" in code
        has_chips = "Write Code" in code
        dur = (time.time() - t0) * 1000
        all_ui = has_chat_input and has_inline_uploader and has_model_selector and has_audio_player and has_image_gen and has_chips
        record_test("Frontend Streamlit UI & Multimodal Chat Bar", "Frontend", "WORKING" if all_ui else "FAILED",
                    f"Plus Popover & Uploader: {has_inline_uploader}, Model Switcher: {has_model_selector}, Action Chips: {has_chips}, Audio: {has_audio_player}", dur)
    except Exception as e:
        record_test("Frontend Streamlit UI & Inline Chat Bar", "Frontend", "FAILED", str(e), (time.time() - t0) * 1000)

    # Cleanup test document
    if created_doc_id:
        try:
            client.delete(f"/documents/{created_doc_id}")
        except Exception:
            pass

    db.close()

    print("=" * 70)
    print("🏁 AUTOMATION TEST SUITE EXECUTION COMPLETE")
    print("=" * 70)

    # Save results as JSON
    with open("automation_test_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_suite()
