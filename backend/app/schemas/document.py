from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    file_size: int
    status: str
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    original_text: Optional[str] = None
    doc_metadata: Dict[str, Any] = Field(default_factory=dict)


class GeneratedDocumentCreate(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_document_id: Optional[str] = None
    format: str = Field("markdown", pattern="^(markdown|txt)$")


class GeneratedDocumentUpdate(BaseModel):
    content: str = Field(..., min_length=1)
    revision_notes: Optional[str] = None


class GeneratedDocumentResponse(BaseModel):
    id: str
    title: str
    content: str
    format: str
    version: int
    source_document_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
