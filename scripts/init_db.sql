-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: stores raw chunks with metadata
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source      TEXT NOT NULL,
    source_type TEXT NOT NULL,              -- 'pdf' | 'web'
    title       TEXT,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Embeddings table: vectors linked to documents
CREATE TABLE IF NOT EXISTS embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   vector(1024),              -- dimension matches BAAI/bge-small-en-v1.5
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for fast ANN search
CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);
CREATE INDEX IF NOT EXISTS documents_source_type_idx ON documents (source_type);

-- Evaluation traces
CREATE TABLE IF NOT EXISTS eval_traces (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query            TEXT NOT NULL,
    answer           TEXT NOT NULL,
    contexts         TEXT[] NOT NULL,
    faithfulness     FLOAT,
    context_recall   FLOAT,
    answer_relevancy FLOAT,
    latency_ms       INTEGER,
    model            TEXT,
    created_at       TIMESTAMPTZ DEFAULT now()
);

-- Auto-update trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();