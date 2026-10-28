from typing import List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str  # "gemini", "grok", "mock"
    is_available: bool
    supports_vision: bool = False
    supports_audio: bool = False
    description: str


class AvailableModelsResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str
    auto_fallback_enabled: bool = True
    active_providers: List[str]
