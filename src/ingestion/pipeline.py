"""
Ingestion pipeline: source → chunks → embeddings → pgvector.
Supports PDF files and web pages (Databricks / Spark / dbt docs).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import psycopg
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

SourceType = Literal["pdf", "web"]


@dataclass
class DocumentChunk:
    source: str
    source_type: SourceType
    title: str
    chunk_text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Unique hash to avoid duplicate ingestion."""
        return hashlib.sha256(
            f"{self.source}:{self.chunk_index}:{self.chunk_text}".encode()
        ).hexdigest()


class TextSplitter:
    """Recursive character splitter with boundary awareness."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    idx = text.rfind(sep, start, end)
                    if idx > start:
                        end = idx + len(sep)
                        break
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if len(c) > 20]


class EmbeddingModel:
    """Mistral embeddings via raw HTTP — zero native dependencies."""

    API_URL = "https://api.mistral.ai/v1/embeddings"
    MODEL = "mistral-embed"  # dimension: 1024

    def __init__(self) -> None:
        from src.utils.config import settings as _settings

        logger.info("loading_embedding_model", model=self.MODEL)
        self._api_key = _settings.mistral_api_key

    def encode(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(4):
            with httpx.Client(timeout=60) as client:
                response = client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.MODEL, "input": texts},
                )
                if response.status_code == 429:
                    wait = 2**attempt  # 1s, 2s, 4s, 8s
                    logger.warning("rate_limit_hit", attempt=attempt, wait=wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        raise RuntimeError("Mistral API rate limit exceeded after 4 attempts")


class PDFLoader:
    def load(self, path: Path) -> tuple[str, str]:
        """Returns (title, full_text)."""
        reader = PdfReader(path)
        title = reader.metadata.title or path.stem
        pages = [page.extract_text() or "" for page in reader.pages]
        return title, "\n\n".join(pages)


class WebLoader:
    """Fetches and parses HTML documentation pages."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def load(self, url: str) -> tuple[str, str]:
        """Returns (title, clean_text)."""
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(
                url, headers={"User-Agent": "RAG-Agent-Portfolio/1.0"}
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()

        title = soup.title.string if soup.title else url
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text(separator="\n", strip=True) if main else ""

        return str(title).strip(), text


class IngestionPipeline:
    """Orchestrates the full ingestion flow."""

    def __init__(self) -> None:
        self.splitter = TextSplitter(settings.chunk_size, settings.chunk_overlap)
        self.embedder = EmbeddingModel()

    def _make_chunks(
        self,
        source: str,
        source_type: SourceType,
        title: str,
        text: str,
    ) -> list[DocumentChunk]:
        raw_chunks = self.splitter.split(text)
        return [
            DocumentChunk(
                source=source,
                source_type=source_type,
                title=title,
                chunk_text=chunk,
                chunk_index=i,
                metadata={"char_count": len(chunk)},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def ingest_pdf(self, path: Path) -> int:
        loader = PDFLoader()
        title, text = loader.load(path)
        chunks = self._make_chunks(str(path), "pdf", title, text)
        return self._store(chunks)

    def ingest_url(self, url: str) -> int:
        loader = WebLoader()
        title, text = loader.load(url)
        chunks = self._make_chunks(url, "web", title, text)
        return self._store(chunks)

    def _store(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        texts = [c.chunk_text for c in chunks]
        vectors = self.embedder.encode(texts)
        stored = 0
        conn_str = settings.postgres_url.replace("+psycopg", "")

        with psycopg.connect(conn_str) as conn, conn.cursor() as cur:
            fingerprints = [c.fingerprint for c in chunks]
            cur.execute(
                "SELECT metadata->>'fingerprint' FROM documents "
                "WHERE metadata->>'fingerprint' = ANY(%s)",
                (fingerprints,),
            )
            existing = {row[0] for row in cur.fetchall()}

            for chunk, vector in zip(chunks, vectors, strict=True):
                if chunk.fingerprint in existing:
                    continue
                doc_id = uuid.uuid4()
                cur.execute(
                    """
                        INSERT INTO documents
                            (id, source, source_type, title, content, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                    (
                        str(doc_id),
                        chunk.source,
                        chunk.source_type,
                        chunk.title,
                        chunk.chunk_text,
                        json.dumps(
                            {
                                "chunk_index": chunk.chunk_index,
                                "fingerprint": chunk.fingerprint,
                                **chunk.metadata,
                            }
                        ),
                    ),
                )
                cur.execute(
                    """
                        INSERT INTO embeddings
                            (document_id, chunk_index, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                    (str(doc_id), chunk.chunk_index, chunk.chunk_text, vector),
                )
                stored += 1

            conn.commit()

        logger.info(
            "ingestion_complete",
            source=chunks[0].source,
            total_chunks=len(chunks),
            stored=stored,
            skipped=len(chunks) - stored,
        )
        return stored
