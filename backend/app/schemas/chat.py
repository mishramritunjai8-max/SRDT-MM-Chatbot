from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    source_type: str = "rag"  # "rag" or "web"
    score: Optional[float] = None
    page: Optional[int] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's prompt or question")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID")
    model: Optional[str] = Field("auto", description="Selected model or 'auto' for smart auto-switching")
    system_prompt: Optional[str] = Field(None, description="Custom system instruction override")
    use_rag: bool = Field(False, description="Enable Retrieval-Augmented Generation over uploaded documents")
    document_id: Optional[str] = Field(None, description="Filter RAG to a specific document ID")
    document_ids: Optional[List[str]] = Field(None, description="Filter RAG to multiple specific document IDs")
    use_web_search: bool = Field(False, description="Enable live web search grounding")
    emotional_mode: bool = Field(False, description="Enable empathetic personal emotional support mode")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    switched_from: Optional[str] = None
    switch_reason: Optional[str] = None
    sources: List[SourceItem] = Field(default_factory=list)
