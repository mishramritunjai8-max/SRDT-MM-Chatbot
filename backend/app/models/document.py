from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.models.base import Base, generate_uuid, get_utc_now


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(String(50), default="pending")  # pending, processed, error
    original_text = Column(Text, nullable=True)
    doc_metadata = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    # Stored as list of floats (JSON/Vector compatible)
    embedding = Column(JSON, nullable=True)
    chunk_metadata = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=get_utc_now)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class GeneratedDocument(Base):
    __tablename__ = "generated_documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    source_document_id = Column(String(36), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String(20), default="markdown")  # markdown, txt
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)
