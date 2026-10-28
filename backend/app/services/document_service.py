import os
import io
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from pypdf import PdfReader
import docx

from backend.app.core.config import settings
from backend.app.models.document import Document, DocumentChunk, GeneratedDocument
from backend.app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks recursively by paragraphs, sentences, or words."""
    if not text:
        return []

    # If text fits in one chunk
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        # Try to break on a sentence boundary or newline if not at the end
        if end < text_length:
            boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
        if start >= text_length or end >= text_length:
            break

    return chunks


class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        self.embedder = EmbeddingService()

    def extract_text(self, file_bytes: bytes, filename: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts raw text and per-page/section metadata from PDF, DOCX, TXT.
        Returns: (full_text, pages_metadata)
        """
        ext = os.path.splitext(filename)[1].lower()
        full_text = ""
        pages_metadata: List[Dict[str, Any]] = []

        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                full_text += f"\n[Page {i + 1}]\n" + page_text
                pages_metadata.append({"page": i + 1, "char_len": len(page_text)})

        elif ext in [".docx", ".doc"]:
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            pages_metadata.append({"paragraphs": len(paragraphs)})

        elif ext in [".txt", ".md", ".csv"]:
            full_text = file_bytes.decode("utf-8", errors="replace")
            pages_metadata.append({"lines": len(full_text.splitlines())})

        else:
            raise ValueError(f"Unsupported document format: '{ext}'. Allowed formats: .pdf, .docx, .txt, .md")

        return full_text.strip(), pages_metadata

    def ingest_document(self, filename: str, content_type: str, file_bytes: bytes) -> Document:
        """
        Ingest a document: parse text, save record, generate chunks and embeddings.
        """
        doc_id = str(uuid.uuid4())
        full_text, metadata = self.extract_text(file_bytes, filename)

        doc = Document(
            id=doc_id,
            filename=filename,
            content_type=content_type,
            file_size=len(file_bytes),
            status="processing",
            original_text=full_text,
            doc_metadata={"pages": metadata}
        )
        self.db.add(doc)
        self.db.commit()

        try:
            # Chunking
            chunks = split_text_into_chunks(
                full_text,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )

            # Generate embeddings and store chunks
            for idx, chunk_str in enumerate(chunks):
                emb = self.embedder.embed_text(chunk_str)
                chunk_record = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    chunk_index=idx,
                    chunk_text=chunk_str,
                    embedding=emb,
                    chunk_metadata={"chunk_index": idx, "filename": filename}
                )
                self.db.add(chunk_record)

            doc.status = "processed"
            self.db.commit()
            self.db.refresh(doc)
            return doc

        except Exception as e:
            logger.error(f"Error chunking/embedding document {doc_id}: {e}")
            doc.status = "error"
            self.db.commit()
            raise

    def get_document(self, document_id: str) -> Optional[Document]:
        return self.db.query(Document).filter(Document.id == document_id).first()

    def list_documents(self) -> List[Document]:
        return self.db.query(Document).order_by(Document.created_at.desc()).all()

    def delete_document(self, document_id: str) -> bool:
        doc = self.get_document(document_id)
        if not doc:
            return False
        self.db.delete(doc)
        self.db.commit()
        return True

    def reprocess_document(self, document_id: str) -> Document:
        """Re-chunks and re-embeds an existing document."""
        doc = self.get_document(document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found.")

        # Clear existing chunks
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        self.db.commit()

        # Re-split and embed
        chunks = split_text_into_chunks(
            doc.original_text or "",
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        for idx, chunk_str in enumerate(chunks):
            emb = self.embedder.embed_text(chunk_str)
            chunk_record = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=idx,
                chunk_text=chunk_str,
                embedding=emb,
                chunk_metadata={"chunk_index": idx, "filename": doc.filename}
            )
            self.db.add(chunk_record)

        doc.status = "processed"
        self.db.commit()
        self.db.refresh(doc)
        return doc

    # Generated Document versioning
    def create_generated_document(
        self,
        title: str,
        content: str,
        source_document_id: Optional[str] = None,
        doc_format: str = "markdown"
    ) -> GeneratedDocument:
        gen_doc = GeneratedDocument(
            id=str(uuid.uuid4()),
            source_document_id=source_document_id,
            title=title,
            content=content,
            format=doc_format,
            version=1
        )
        self.db.add(gen_doc)
        self.db.commit()
        self.db.refresh(gen_doc)
        return gen_doc

    def update_generated_document(
        self,
        document_id: str,
        new_content: str,
        revision_notes: Optional[str] = None
    ) -> GeneratedDocument:
        gen_doc = self.db.query(GeneratedDocument).filter(GeneratedDocument.id == document_id).first()
        if not gen_doc:
            raise ValueError(f"Generated document {document_id} not found.")

        gen_doc.content = new_content
        gen_doc.version += 1
        self.db.commit()
        self.db.refresh(gen_doc)
        return gen_doc
