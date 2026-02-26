"""PDF ingestion pipeline: parse, chunk, and prepare for embedding."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, get_settings


@dataclass
class PaperMetadata:
    """Extracted paper metadata."""

    paper_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    abstract: str = ""
    filename: str = ""
    total_pages: int = 0


@dataclass
class TextChunk:
    """A chunk of text with source metadata."""

    chunk_id: str
    paper_id: str
    text: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _generate_paper_id(filepath: Path) -> str:
    """Generate a stable ID from file content hash."""
    h = hashlib.md5(filepath.read_bytes()).hexdigest()[:12]
    return f"paper_{h}"


def _clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Fix hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_metadata(doc: fitz.Document, filepath: Path) -> PaperMetadata:
    """Extract metadata from a PDF document."""
    meta = doc.metadata or {}
    paper_id = _generate_paper_id(filepath)

    title = meta.get("title", "")
    if not title or title.strip() == "":
        # Try first page first lines as fallback
        first_page = doc[0].get_text()
        lines = [l.strip() for l in first_page.split("\n") if l.strip()]
        title = lines[0] if lines else filepath.stem

    authors_raw = meta.get("author", "")
    authors = [a.strip() for a in authors_raw.split(",") if a.strip()] if authors_raw else []

    # Try to extract DOI from first 2 pages
    doi = ""
    for page_num in range(min(2, len(doc))):
        page_text = doc[page_num].get_text()
        doi_match = re.search(r"(10\.\d{4,}/[^\s]+)", page_text)
        if doi_match:
            doi = doi_match.group(1).rstrip(".,;)")
            break

    # Try to extract abstract
    abstract = ""
    first_pages = ""
    for page_num in range(min(2, len(doc))):
        first_pages += doc[page_num].get_text() + "\n"

    abs_match = re.search(
        r"(?:Abstract|ABSTRACT|Summary|SUMMARY)[:\s]*\n?(.*?)(?:\n\s*\n|\n(?:Introduction|INTRODUCTION|Keywords|KEYWORDS|1\.|1\s))",
        first_pages,
        re.DOTALL | re.IGNORECASE,
    )
    if abs_match:
        abstract = _clean_text(abs_match.group(1))

    return PaperMetadata(
        paper_id=paper_id,
        title=title.strip(),
        authors=authors,
        doi=doi,
        abstract=abstract,
        filename=filepath.name,
        total_pages=len(doc),
    )


def extract_text_by_page(doc: fitz.Document) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF."""
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        cleaned = _clean_text(text)
        if cleaned:
            pages.append((page_num + 1, cleaned))
    return pages


def chunk_text(
    pages: list[tuple[int, str]],
    paper_id: str,
    settings: Settings | None = None,
) -> list[TextChunk]:
    """Split page texts into overlapping chunks with metadata."""
    if settings is None:
        settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[TextChunk] = []
    chunk_idx = 0

    for page_num, page_text in pages:
        splits = splitter.split_text(page_text)
        for split_text in splits:
            chunk_id = f"{paper_id}_chunk_{chunk_idx:04d}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=split_text,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    metadata={
                        "paper_id": paper_id,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                    },
                )
            )
            chunk_idx += 1

    return chunks


def ingest_pdf(filepath: str | Path, settings: Settings | None = None) -> tuple[PaperMetadata, list[TextChunk]]:
    """
    Full ingestion pipeline for a single PDF.

    Returns:
        Tuple of (paper_metadata, text_chunks)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    if settings is None:
        settings = get_settings()

    doc = fitz.open(str(filepath))
    try:
        metadata = extract_metadata(doc, filepath)
        pages = extract_text_by_page(doc)
        chunks = chunk_text(pages, metadata.paper_id, settings)
    finally:
        doc.close()

    return metadata, chunks


def ingest_markdown(filepath: str | Path, settings: Settings | None = None) -> tuple[PaperMetadata, list[TextChunk]]:
    """
    Ingest a markdown file (e.g., PubMed abstract saved as .md).

    Returns:
        Tuple of (paper_metadata, text_chunks)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if settings is None:
        settings = get_settings()

    text = filepath.read_text(encoding="utf-8")

    # Try to parse YAML frontmatter
    title, authors, doi, abstract, pmid = "", [], "", "", ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            import yaml
            try:
                meta = yaml.safe_load(parts[1])
                title = meta.get("title", "")
                authors = meta.get("authors", [])
                doi = meta.get("doi", "")
                pmid = meta.get("pmid", "")
            except Exception:
                pass
            text = parts[2]

    # Extract abstract from body if not in frontmatter
    abs_match = re.search(r'## Abstract\s*\n(.+?)(?:\n## |$)', text, re.DOTALL)
    if abs_match:
        abstract = abs_match.group(1).strip()

    if not title:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        title = lines[0].lstrip("# ") if lines else filepath.stem

    paper_id = f"paper_{pmid}" if pmid else _generate_paper_id(filepath)

    metadata = PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=authors if isinstance(authors, list) else [authors],
        doi=doi,
        abstract=abstract,
        filename=filepath.name,
        total_pages=1,
    )

    body = _clean_text(text)
    chunks = chunk_text([(1, body)], metadata.paper_id, settings)
    return metadata, chunks


def ingest_file(filepath: str | Path, settings: Settings | None = None) -> tuple[PaperMetadata, list[TextChunk]]:
    """Ingest a PDF or markdown file."""
    filepath = Path(filepath)
    if filepath.suffix.lower() == ".md":
        return ingest_markdown(filepath, settings)
    else:
        return ingest_pdf(filepath, settings)


def ingest_directory(
    dir_path: str | Path,
    settings: Settings | None = None,
) -> list[tuple[PaperMetadata, list[TextChunk]]]:
    """Ingest all PDFs and markdown files in a directory."""
    dir_path = Path(dir_path)
    results = []

    files = sorted(
        list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.md"))
    )
    if not files:
        raise FileNotFoundError(f"No PDF or markdown files found in {dir_path}")

    for f in files:
        try:
            result = ingest_file(f, settings)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Failed to ingest {f.name}: {e}")

    return results
