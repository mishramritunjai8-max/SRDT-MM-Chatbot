from backend.app.api.models import router as models_router
from backend.app.api.chat import router as chat_router
from backend.app.api.documents import router as documents_router
from backend.app.api.multimodal import router as multimodal_router

__all__ = ["models_router", "chat_router", "documents_router", "multimodal_router"]
