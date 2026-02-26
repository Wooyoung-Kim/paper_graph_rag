"""ChromaDB vector store for semantic search over paper chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import get_settings


class VectorStore:
    """ChromaDB-based vector store for paper chunk embeddings."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "paper_chunks",
    ):
        settings = get_settings()
        if persist_dir is None:
            settings.paths.ensure_dirs()
            persist_dir = settings.paths.chroma_dir

        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding.model_name
        )

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add text chunks to the vector store."""
        if not chunk_ids:
            return

        # ChromaDB doesn't allow duplicate IDs, filter existing
        existing = set()
        try:
            result = self._collection.get(ids=chunk_ids)
            existing = set(result["ids"])
        except Exception:
            pass

        new_ids = []
        new_texts = []
        new_metas = []
        for i, cid in enumerate(chunk_ids):
            if cid not in existing:
                new_ids.append(cid)
                new_texts.append(texts[i])
                if metadatas:
                    new_metas.append(metadatas[i])

        if not new_ids:
            return

        # Add in batches of 100
        batch_size = 100
        for start in range(0, len(new_ids), batch_size):
            end = start + batch_size
            batch_metas = new_metas[start:end] if new_metas else None
            self._collection.add(
                ids=new_ids[start:end],
                documents=new_texts[start:end],
                metadatas=batch_metas,
            )

    def search(
        self,
        query: str,
        n_results: int = 10,
        paper_id: str | None = None,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Search query text
            n_results: Number of results
            paper_id: Optional filter by paper ID
            where: Optional metadata filter dict

        Returns:
            List of dicts with 'id', 'text', 'metadata', 'distance'
        """
        query_params: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, self.count) if self.count > 0 else 1,
        }

        if paper_id:
            query_params["where"] = {"paper_id": paper_id}
        elif where:
            query_params["where"] = where

        if self.count == 0:
            return []

        results = self._collection.query(**query_params)

        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
            })

        return items

    def delete_paper(self, paper_id: str) -> None:
        """Delete all chunks for a specific paper."""
        self._collection.delete(where={"paper_id": paper_id})

    def get_paper_ids(self) -> list[str]:
        """Get all unique paper IDs in the store."""
        if self.count == 0:
            return []

        # Get all metadata
        result = self._collection.get(include=["metadatas"])
        paper_ids = set()
        for meta in result["metadatas"]:
            if "paper_id" in meta:
                paper_ids.add(meta["paper_id"])
        return sorted(paper_ids)
