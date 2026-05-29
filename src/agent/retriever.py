"""
Vector retriever with pgvector cosine similarity search.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from src.ingestion.pipeline import EmbeddingModel
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    title: str
    score: float
    chunk_index: int
    metadata: dict


class VectorRetriever:
    """Retrieves relevant chunks from pgvector using cosine similarity."""

    def __init__(self, top_k: int = 6) -> None:
        self.top_k = top_k
        self._embedder = EmbeddingModel()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k most similar chunks for a query.

        Args:
            query: User question
            top_k: Override default top_k
            source_filter: Optional SQL LIKE pattern on source (e.g. '%databricks%')
        """
        k = top_k or self.top_k
        query_vector = self._embedder.encode([query])[0]
        conn_str = settings.postgres_url.replace("+psycopg", "")

        with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
            if source_filter:
                cur.execute(
                    """
                        SELECT e.chunk_text, d.source, d.title,
                               1 - (e.embedding <=> %s::vector) AS score,
                               e.chunk_index, d.metadata
                        FROM embeddings e
                        JOIN documents d ON d.id = e.document_id
                        WHERE d.source LIKE %s
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                        """,
                    (query_vector, source_filter, query_vector, k),
                )
            else:
                cur.execute(
                    """
                        SELECT e.chunk_text, d.source, d.title,
                               1 - (e.embedding <=> %s::vector) AS score,
                               e.chunk_index, d.metadata
                        FROM embeddings e
                        JOIN documents d ON d.id = e.document_id
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                        """,
                    (query_vector, query_vector, k),
                )
            rows = cur.fetchall()

        chunks = [
            RetrievedChunk(
                content=row[0],
                source=row[1],
                title=row[2],
                score=float(row[3]),
                chunk_index=row[4],
                metadata=row[5] or {},
            )
            for row in rows
        ]

        # Filter low-relevance chunks
        chunks = [c for c in chunks if c.score >= 0.30]

        logger.info(
            "retrieval_done",
            query_preview=query[:60],
            retrieved=len(chunks),
            top_score=round(chunks[0].score, 3) if chunks else 0,
        )
        return chunks
