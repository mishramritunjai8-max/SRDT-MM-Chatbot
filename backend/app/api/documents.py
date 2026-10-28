from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.schemas.document import (
    DocumentResponse,
    DocumentDetailResponse,
    GeneratedDocumentCreate,
    GeneratedDocumentUpdate,
    GeneratedDocumentResponse
)
from backend.app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Document Management & RAG"])


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and parse a document (PDF, DOCX, TXT), split into chunks,
    generate embeddings, and store in database for RAG.
    """
    allowed_extensions = [".pdf", ".docx", ".doc", ".txt", ".md"]
    filename = file.filename or "uploaded_document"
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported formats: {', '.join(allowed_extensions)}"
        )

    try:
        content = await file.read()
        service = DocumentService(db)
        doc = service.ingest_document(
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            file_bytes=content
        )
        return DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            content_type=doc.content_type,
            file_size=doc.file_size,
            status=doc.status,
            chunk_count=len(doc.chunks),
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents with status and chunk count."""
    service = DocumentService(db)
    docs = service.list_documents()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            content_type=d.content_type,
            file_size=d.file_size,
            status=d.status,
            chunk_count=len(d.chunks),
            created_at=d.created_at,
            updated_at=d.updated_at
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Retrieve metadata and extracted text for a specific document."""
    service = DocumentService(db)
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetailResponse(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        file_size=doc.file_size,
        status=doc.status,
        chunk_count=len(doc.chunks),
        original_text=doc.original_text,
        doc_metadata=doc.doc_metadata or {},
        created_at=doc.created_at,
        updated_at=doc.updated_at
    )


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and its cascading chunk embeddings."""
    service = DocumentService(db)
    success = service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document and associated embeddings successfully deleted."}


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
def reprocess_document(document_id: str, db: Session = Depends(get_db)):
    """Reprocess document: re-chunks and updates embeddings."""
    service = DocumentService(db)
    try:
        doc = service.reprocess_document(document_id)
        return DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            content_type=doc.content_type,
            file_size=doc.file_size,
            status=doc.status,
            chunk_count=len(doc.chunks),
            created_at=doc.created_at,
            updated_at=doc.updated_at
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create", response_model=GeneratedDocumentResponse, status_code=status.HTTP_201_CREATED)
def create_generated_document(payload: GeneratedDocumentCreate, db: Session = Depends(get_db)):
    """Create a new AI-generated downloadable document (Markdown or TXT)."""
    service = DocumentService(db)
    gen_doc = service.create_generated_document(
        title=payload.title,
        content=payload.content,
        source_document_id=payload.source_document_id,
        doc_format=payload.format
    )
    return gen_doc


@router.put("/generated/{document_id}", response_model=GeneratedDocumentResponse)
def update_generated_document(
    document_id: str,
    payload: GeneratedDocumentUpdate,
    db: Session = Depends(get_db)
):
    """Update a generated document with revisions, incrementing version number."""
    service = DocumentService(db)
    try:
        return service.update_generated_document(
            document_id=document_id,
            new_content=payload.content,
            revision_notes=payload.revision_notes
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
