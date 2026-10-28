from fastapi import APIRouter
from backend.app.schemas.models import AvailableModelsResponse
from backend.app.services.provider_router import ProviderRouter

router = APIRouter(prefix="/models", tags=["Models & Providers"])
provider_router = ProviderRouter()


@router.get("/available", response_model=AvailableModelsResponse)
def get_available_models():
    """
    Returns the list of configured models across Google Gemini and xAI Grok,
    including real-time availability status and auto-fallback capabilities.
    """
    return provider_router.get_available_models()
