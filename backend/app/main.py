from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.db.session import init_db, get_db
from backend.app.services.provider_router import ProviderRouter
from backend.app.api.models import router as models_router
from backend.app.api.chat import router as chat_router
from backend.app.api.documents import router as documents_router
from backend.app.api.multimodal import router as multimodal_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Multimodal AI Chatbot MVP Backend with Gemini, Grok, and RAG vector search.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(multimodal_router)


@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Health check endpoint returning system status, DB connectivity,
    and available AI providers.
    """
    db_status = "healthy"
    db_type = "postgresql" if settings.DATABASE_URL.startswith("postgres") else "sqlite"
    has_vector = False

    try:
        if db_type == "postgresql":
            result = db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).fetchone()
            has_vector = result is not None
        else:
            db.execute(text("SELECT 1")).fetchone()
            has_vector = True  # Python cosine similarity adapter active
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    provider_router = ProviderRouter()
    models_info = provider_router.get_available_models()

    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "database": {
            "status": db_status,
            "type": db_type,
            "vector_search_ready": has_vector
        },
        "providers": {
            "gemini": "configured" if "gemini" in models_info.active_providers else "missing_key",
            "grok": "configured" if "grok" in models_info.active_providers else "missing_key",
            "mock": "always_ready"
        },
        "default_model": models_info.default_model,
        "auto_fallback_enabled": True
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )
