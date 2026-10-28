# Multimodal AI Chatbot MVP (Grok & Gemini with Auto-Switching)

A full-stack, beginner-friendly **Multimodal AI Chatbot** combining conversational AI, multimodal inputs/outputs (text, images, documents, audio), live web search grounding, document-based Retrieval-Augmented Generation (RAG), and an empathetic personal emotional support mode.

Built natively with **FastAPI**, **PostgreSQL** (with pgvector and seamless zero-setup SQLite fallback), and **Streamlit**.

---

## 🌟 Key Features

1. **Dual AI Providers with Smart Auto-Switching:**
   - **Google Gemini** (`gemini-2.5-flash` / `gemini-1.5-flash`): Ultra-fast multimodal chat, vision, audio understanding, and vector embeddings.
   - **xAI Grok** (`grok-2`, `grok-2-vision-1212` via OpenAI-compatible standard): High-reasoning chat, coding nuance, and high-resolution vision.
   - **Auto-Switching / Failover Engine:** When `"auto"` is selected (or when a preferred provider encounters rate limits, HTTP 429, or network outages), the system automatically routes to the alternative healthy provider and displays a transparent notification.
   - **Offline Mock Provider:** When neither key is supplied, a deterministic local provider engages so local grading, testing, and demonstrations work flawlessly without paid keys.

2. **Multimodal Input & Output:**
   - **Text:** General conversational chat with session history and clear-chat control.
   - **Vision (Images):** Upload JPG, JPEG, and PNG images and ask analytical questions.
   - **Documents:** Ingest PDF, DOCX, TXT, and Markdown files for grounded RAG answers.
   - **Audio (TTS & STT):** Convert assistant responses to speech (`gTTS`) with in-browser audio player.

3. **Retrieval-Augmented Generation (RAG):**
   - Recursive character chunking with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`.
   - Vector similarity search with hybrid adapter:
     - **Primary:** PostgreSQL + `pgvector` (`vector_cosine_ops`).
     - **Fallback / Local Dev:** SQLite with in-process vector cosine similarity.
   - Grounded prompt enforcement: Answers strictly from context, explicitly admits missing information, and provides chunk & document citations.

4. **Live Web Search Grounding:**
   - Real-time web retrieval via DuckDuckGo (privacy-friendly, zero-cost, no API key required).
   - Clickable source URLs and synthesized grounded summaries.

5. **Document Creation & Versioning:**
   - Export AI-generated responses into versioned Markdown or TXT documents.
   - Manage, reprocess, and delete documents with cascading chunk cleanup.

6. **Personal Emotional Assistance:**
   - Empathetic reflection mode designed for calm conversations and gentle coping strategies.
   - Non-clinical boundaries (no diagnosis, no prescription).
   - Strict crisis detection filter that surfaces verified emergency resources (e.g., 988 Suicide & Crisis Lifeline).
   - Ephemeral session memory to protect user privacy.

---

## 🏗️ Architecture & Data Flow

```text
[ Browser / User ]
        ▲
        │ HTTP / Web UI
        ▼
[ Streamlit Frontend (:8501) ]
        ▲
        │ REST JSON API
        ▼
[ FastAPI Backend (:8000) ]
   ├── /health
   ├── /models/available
   ├── /chat  (Conversational, RAG, Web Search, Emotional Mode)
   ├── /documents/upload, /documents, /documents/{id}
   └── /multimodal/vision, /multimodal/synthesize, /multimodal/transcribe
        ▲
        ├───────────────┬────────────────┬────────────────┐
        ▼               ▼                ▼                ▼
[ Provider Router ] [ RAG Engine ]  [ Web Search ]  [ Database ]
  ├── Grok (xAI)      └── Chunker     └── DuckDuckGo  ├── PostgreSQL + pgvector
  ├── Gemini (Google) └── Embeddings                  └── SQLite (Local fallback)
  └── Mock Fallback   └── Cosine Sim
```

---

## 📁 Project Structure

```text
MM-Chatbot/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry point & health check
│   │   ├── api/
│   │   │   ├── chat.py                # POST /chat endpoint
│   │   │   ├── models.py              # GET /models/available endpoint
│   │   │   ├── documents.py           # Document upload, listing, CRUD, versioning
│   │   │   └── multimodal.py          # Vision and audio endpoints
│   │   ├── core/
│   │   │   └── config.py              # Pydantic environment configuration
│   │   ├── db/
│   │   │   ├── session.py             # SQLAlchemy engine & session dependency
│   │   │   └── vector_adapter.py      # Hybrid pgvector & SQLite cosine adapter
│   │   ├── models/
│   │   │   ├── base.py                # SQLAlchemy declarative base
│   │   │   ├── document.py            # Document, DocumentChunk, GeneratedDocument
│   │   │   └── chat.py                # ChatSession, ChatMessage
│   │   ├── schemas/
│   │   │   ├── chat.py                # ChatRequest & ChatResponse schemas
│   │   │   ├── models.py              # Model discovery schemas
│   │   │   └── document.py            # Document CRUD schemas
│   │   ├── services/
│   │   │   ├── provider_router.py     # Grok & Gemini auto-switching engine
│   │   │   ├── gemini_service.py      # Google Gemini client
│   │   │   ├── grok_service.py        # xAI Grok client
│   │   │   ├── mock_service.py        # Deterministic offline mock
│   │   │   ├── chat_service.py        # Conversation orchestration
│   │   │   ├── document_service.py    # Document parsers & versioning
│   │   │   ├── rag_service.py         # Grounded vector retrieval
│   │   │   ├── embedding_service.py   # Embedding generator
│   │   │   ├── web_search_service.py  # DuckDuckGo search integration
│   │   │   ├── vision_service.py      # Image analysis service
│   │   │   ├── audio_service.py       # Speech synthesis (TTS) & transcription
│   │   │   └── emotional_support_service.py # Empathetic boundaries & crisis filter
│   │   └── utils/
│   │       └── helpers.py             # File size and snippet helpers
│   ├── tests/
│   │   ├── conftest.py                # Test fixtures & SQLite test DB
│   │   ├── test_health.py             # Health check endpoint tests
│   │   ├── test_models.py             # Model availability tests
│   │   ├── test_chat.py               # Chat & failover tests
│   │   ├── test_documents.py          # Document CRUD & versioning tests
│   │   ├── test_rag.py                # Vector search & grounded citation tests
│   │   └── test_multimodal.py         # Vision & audio tests
│   └── requirements.txt
├── frontend/
│   ├── app.py                         # Streamlit user interface
│   ├── api_client.py                  # HTTP REST client to FastAPI
│   └── requirements.txt
├── scripts/
│   └── init_db.sql                    # PostgreSQL schema with pgvector
├── .env.example                       # Template configuration
├── .gitignore
├── pytest.ini
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.13)
- Optional: **PostgreSQL** with `pgvector` extension (If not installed, the application automatically uses the built-in SQLite vector adapter with zero configuration needed).

### 2. Environment Setup

Create and activate a virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install backend and frontend dependencies:

```powershell
pip install -r backend/requirements.txt -r frontend/requirements.txt
```

### 3. Configuration (`.env`)

Copy `.env.example` to `.env`:

```powershell
cp .env.example .env
```

Configure your API keys in `.env`:

```env
# AI Provider Keys (Leave blank to run in offline Mock mode)
GEMINI_API_KEY="your-google-gemini-api-key"
GROK_API_KEY="your-xai-grok-api-key"

# Database Configuration
# Primary (PostgreSQL + pgvector):
# DATABASE_URL="postgresql+psycopg://postgres:password@localhost:5432/mm_chatbot"
# Fallback (Zero-setup local SQLite):
DATABASE_URL="sqlite:///./mm_chatbot.db"
```

---

## 🏃 Running the Application

### 1. Start the FastAPI Backend

```powershell
.\.venv\Scripts\uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive API Docs (Swagger UI): **http://127.0.0.1:8000/docs**
- Health Endpoint: **http://127.0.0.1:8000/health**

### 2. Start the Streamlit Frontend

In a separate terminal:

```powershell
.\.venv\Scripts\streamlit run frontend/app.py
```
- Open your browser at: **http://localhost:8501**

---

## 🧪 Running Automated Tests

Run the full automated test suite using `pytest`:

```powershell
.\.venv\Scripts\pytest
```

---

## 🛡️ Safety, Privacy & Disclaimers

1. **Emotional Support Mode:** This feature is provided for general reflection and coping tips. It is **not** licensed medical therapy, psychiatric advice, or crisis counseling. In an emergency, immediate help is available:
   - **USA / Canada:** Call or text **988**
   - **UK:** Call **111** or Samaritans at **116 123**
   - **India:** Call **14416** (Tele-MANAS) or **112**
   - **International:** Visit [findahelpline.com](https://findahelpline.com/)
2. **Data Retention:** Personal emotional chats are ephemeral and not saved to message history by default.
3. **Document Privacy:** Uploaded documents are parsed locally and stored in your database; they are never shared across sessions.
