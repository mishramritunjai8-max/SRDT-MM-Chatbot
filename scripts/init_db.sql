-- Database Initialization Script for Multimodal AI Chatbot (PostgreSQL + pgvector)

-- 1. Enable the vector extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents table (Original uploaded documents)
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    original_text TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Document chunks table (For RAG vector search)
-- Default embedding dimension 768 (suitable for Gemini text-embedding-004 and standard models)
CREATE TABLE IF NOT EXISTS document_chunks (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast document chunks lookup
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);

-- Optional HNSW or IVFFlat index for vector cosine similarity
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- 4. Generated documents table (AI-created documents & versioning)
CREATE TABLE IF NOT EXISTS generated_documents (
    id VARCHAR(36) PRIMARY KEY,
    source_document_id VARCHAR(36) REFERENCES documents(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    format VARCHAR(20) DEFAULT 'markdown',
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Chat sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(36) PRIMARY KEY,
    session_mode VARCHAR(50) DEFAULT 'general',
    title VARCHAR(255) DEFAULT 'New Conversation',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. Chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    model_used VARCHAR(100),
    switched_from VARCHAR(100),
    switch_reason TEXT,
    sources JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages(session_id);
