import math
from typing import List, Tuple, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.core.config import settings
from backend.app.models.document import DocumentChunk


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two numeric vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class VectorRetrievalAdapter:
    """
    Hybrid Vector Retrieval Adapter.
    - PostgreSQL + pgvector: executes hardware-accelerated SQL vector distance.
    - Fallback (SQLite / Non-pgvector): executes in-process numpy cosine similarity.
    """

    def __init__(self, db: Session):
        self.db = db
        self.is_postgres = settings.DATABASE_URL.startswith("postgres")
        self.has_pgvector = False
        if self.is_postgres:
            self._check_pgvector()

    def _check_pgvector(self) -> None:
        try:
            result = self.db.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).fetchone()
            self.has_pgvector = result is not None
        except Exception:
            self.has_pgvector = False

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        document_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the top-k most similar chunks for a query embedding across single, multiple, or all documents.
        Returns a list of tuples: (DocumentChunk, similarity_score)
        """
        if not query_embedding:
            return []

        # 1. Native PostgreSQL + pgvector retrieval
        if self.is_postgres and self.has_pgvector:
            try:
                emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
                params = {"query_emb": emb_str, "top_k": top_k}
                if document_ids:
                    doc_filter = "AND document_id = ANY(:doc_ids)"
                    params["doc_ids"] = document_ids
                elif document_id:
                    doc_filter = "AND document_id = :doc_id"
                    params["doc_id"] = document_id
                else:
                    doc_filter = ""

                sql = f"""
                    SELECT id, document_id, chunk_index, chunk_text, metadata, created_at,
                           1 - (embedding <=> :query_emb::vector) AS similarity
                    FROM document_chunks
                    WHERE embedding IS NOT NULL {doc_filter}
                    ORDER BY embedding <=> :query_emb::vector ASC
                    LIMIT :top_k
                """

                rows = self.db.execute(text(sql), params).fetchall()
                results = []
                for row in rows:
                    chunk = self.db.query(DocumentChunk).filter(DocumentChunk.id == row[0]).first()
                    if chunk:
                        results.append((chunk, float(row[6])))
                return results
            except Exception:
                # If pgvector execution fails, fall back to in-process scoring
                pass

        # 2. In-process fallback retrieval (SQLite & Non-pgvector)
        query = self.db.query(DocumentChunk)
        if document_ids:
            query = query.filter(DocumentChunk.document_id.in_(document_ids))
        elif document_id:
            query = query.filter(DocumentChunk.document_id == document_id)
        
        all_chunks = query.all()
        scored_chunks: List[Tuple[DocumentChunk, float]] = []

        for chunk in all_chunks:
            if chunk.embedding and isinstance(chunk.embedding, list):
                score = cosine_similarity(query_embedding, chunk.embedding)
                scored_chunks.append((chunk, score))

        # Sort descending by cosine similarity score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
