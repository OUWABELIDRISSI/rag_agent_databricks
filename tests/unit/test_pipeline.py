"""Unit tests for the ingestion pipeline."""

import pytest
from src.ingestion.pipeline import TextSplitter, DocumentChunk


def test_text_splitter_short_text():
    """Text shorter than chunk_size should not be split."""
    splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
    text = "This is a short text."
    chunks = splitter.split(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_text_splitter_long_text():
    """Long text should be split into multiple chunks."""
    splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
    text = "word " * 100  # 500 chars
    chunks = splitter.split(text)
    assert len(chunks) > 1


def test_text_splitter_overlap():
    """Long text with spaces should be split into multiple chunks."""
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    text = ("hello world " * 20).strip()  # texte avec espaces
    chunks = splitter.split(text)
    assert len(chunks) > 1


def test_document_chunk_fingerprint():
    """Same chunk should always produce same fingerprint."""
    chunk = DocumentChunk(
        source="https://docs.databricks.com",
        source_type="web",
        title="Delta Lake",
        chunk_text="Delta Lake is a storage layer.",
        chunk_index=0,
    )
    assert chunk.fingerprint == chunk.fingerprint
    assert len(chunk.fingerprint) == 64  # SHA256 hex


def test_document_chunk_different_fingerprints():
    """Different chunks should produce different fingerprints."""
    chunk1 = DocumentChunk(
        source="https://docs.databricks.com",
        source_type="web",
        title="Delta Lake",
        chunk_text="Delta Lake is a storage layer.",
        chunk_index=0,
    )
    chunk2 = DocumentChunk(
        source="https://docs.databricks.com",
        source_type="web",
        title="Delta Lake",
        chunk_text="Different content here.",
        chunk_index=1,
    )
    assert chunk1.fingerprint != chunk2.fingerprint