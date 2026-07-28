-- ============================================================
-- Run this entire script in Supabase SQL Editor
-- Project: RAG Chatbot - pgvector setup
-- ============================================================

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create vectors table
create table if not exists document_vectors (
  id        bigserial primary key,
  content   text          not null,
  embedding vector(384)           -- 384 dims = all-MiniLM-L6-v2
);

-- 3. IVFFlat index for fast L2 search
create index if not exists document_vectors_embedding_idx
  on document_vectors
  using ivfflat (embedding vector_l2_ops)
  with (lists = 100);

-- 4. RPC function used by search_similar()
create or replace function match_documents(
  query_embedding vector(384),
  match_count     int
)
returns table (
  id      bigint,
  content text,
  score   float
)
language sql stable
as $$
  select
    id,
    content,
    embedding <-> query_embedding as score
  from document_vectors
  order by embedding <-> query_embedding
  limit match_count;
$$;
