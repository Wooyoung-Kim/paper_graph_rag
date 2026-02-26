"""RAG query engine with graph-enhanced retrieval."""

from __future__ import annotations

import os
from typing import Any

from config import LLMConfig, get_settings
from extract import extract_from_chunk, _get_llm
from graph import KnowledgeGraph
from vectorstore import VectorStore


# ============================================================================
# RAG Prompt
# ============================================================================

RAG_SYSTEM_PROMPT = """You are a biomedical research assistant with access to a knowledge base of research papers.
Answer the user's question based on the provided context. Always:

1. Cite specific papers when making claims (use paper IDs)
2. Mention relevant entities and how they connect
3. If the context is insufficient, clearly state what is missing
4. Be precise with scientific terminology

When discussing relationships between entities, explain the evidence from the papers.
Respond in the same language as the user's question."""

RAG_USER_PROMPT = """## Retrieved Context

### Relevant Text Chunks
{chunks_context}

### Knowledge Graph Context
{graph_context}

### Related Papers
{related_papers}

---

## Question
{question}

Please answer based on the above context. Cite paper IDs and mention relevant entity connections."""


# ============================================================================
# RAG Engine
# ============================================================================

class RAGEngine:
    """Graph-enhanced RAG query engine."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        llm_config: LLMConfig | None = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

        if llm_config is None:
            llm_config = get_settings().llm
        self.llm_config = llm_config
        self.llm = _get_llm(llm_config)

    def query(
        self,
        question: str,
        n_chunks: int = 8,
        graph_depth: int = 1,
    ) -> dict[str, Any]:
        """
        Perform a graph-enhanced RAG query.

        Steps:
        1. Vector search for relevant chunks
        2. Extract entities from query
        3. Expand context via knowledge graph
        4. Generate answer with full context

        Returns:
            Dict with 'answer', 'sources', 'related_entities', 'graph_paths'
        """
        # Step 1: Vector search
        search_results = self.vector_store.search(question, n_results=n_chunks)

        # Step 2: Extract entities from query for graph lookup
        query_entities = self._extract_query_entities(question)

        # Step 3: Get graph context
        graph_context = self._get_graph_context(query_entities, search_results, graph_depth)

        # Step 4: Build prompt and generate answer
        chunks_context = self._format_chunks(search_results)
        graph_context_str = self._format_graph_context(graph_context)
        related_papers_str = self._format_related_papers(graph_context)

        prompt = RAG_USER_PROMPT.format(
            chunks_context=chunks_context,
            graph_context=graph_context_str,
            related_papers=related_papers_str,
            question=question,
        )

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": self._collect_sources(search_results),
            "related_entities": graph_context.get("entities", []),
            "graph_paths": graph_context.get("paths", []),
            "chunks_used": len(search_results),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_query_entities(self, question: str) -> list[str]:
        """Extract entity names from the query using the knowledge graph."""
        entities = []
        # Check if any known entity names appear in the query
        for node_name in self.knowledge_graph.graph.nodes():
            if node_name.lower() in question.lower():
                entities.append(node_name)

        # Also try LLM extraction for unrecognized entities
        if not entities:
            try:
                result = extract_from_chunk(question, self.llm_config)
                entities = [e.name for e in result.entities]
            except Exception:
                pass

        return entities

    def _get_graph_context(
        self,
        query_entities: list[str],
        search_results: list[dict],
        depth: int,
    ) -> dict[str, Any]:
        """Build graph context from query entities and search results."""
        context: dict[str, Any] = {
            "entities": [],
            "edges": [],
            "paths": [],
            "related_papers": [],
        }

        # Entities from chunks
        chunk_paper_ids = set()
        for result in search_results:
            pid = result.get("metadata", {}).get("paper_id")
            if pid:
                chunk_paper_ids.add(pid)

        # Get entities mentioned in retrieved papers
        paper_entities: set[str] = set()
        for pid in chunk_paper_ids:
            for ent in self.knowledge_graph.get_entities_for_paper(pid):
                paper_entities.add(ent["name"])

        # Combine with query entities
        all_entities = set(query_entities) | paper_entities

        # Get neighbors for query entities
        for entity_name in query_entities:
            neighbors = self.knowledge_graph.get_neighbors(entity_name, depth=depth)
            context["entities"].extend(neighbors.get("neighbors", []))
            context["edges"].extend(neighbors.get("edges", []))

        # Find paths between query entities
        entity_list = list(query_entities)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                path = self.knowledge_graph.find_path(entity_list[i], entity_list[j])
                if path:
                    context["paths"].append({
                        "from": entity_list[i],
                        "to": entity_list[j],
                        "steps": path,
                    })

        # Find related papers
        for pid in chunk_paper_ids:
            related = self.knowledge_graph.find_related_papers(pid, min_shared=1)
            context["related_papers"].extend(related[:3])

        # Deduplicate
        seen_entities = set()
        unique_entities = []
        for ent in context["entities"]:
            name = ent.get("name", "")
            if name not in seen_entities:
                seen_entities.add(name)
                unique_entities.append(ent)
        context["entities"] = unique_entities

        return context

    def _format_chunks(self, search_results: list[dict]) -> str:
        """Format search results for the prompt."""
        if not search_results:
            return "No relevant chunks found."

        parts = []
        for i, result in enumerate(search_results, 1):
            pid = result.get("metadata", {}).get("paper_id", "unknown")
            page = result.get("metadata", {}).get("page", "?")
            text = result.get("text", "")
            distance = result.get("distance", 0)
            parts.append(
                f"**[Chunk {i}]** (paper: {pid}, page: {page}, relevance: {1-distance:.2f})\n{text}"
            )

        return "\n\n".join(parts)

    def _format_graph_context(self, graph_context: dict) -> str:
        """Format graph context for the prompt."""
        parts = []

        edges = graph_context.get("edges", [])
        if edges:
            parts.append("**Entity Relationships:**")
            seen = set()
            for edge in edges[:15]:
                key = (edge["source"], edge["target"])
                if key not in seen:
                    seen.add(key)
                    parts.append(
                        f"- {edge['source']} → {edge['relationship_type']} → {edge['target']}"
                    )

        paths = graph_context.get("paths", [])
        if paths:
            parts.append("\n**Connection Paths:**")
            for path_info in paths:
                steps = path_info["steps"]
                path_str = " → ".join(
                    f"{s['source']} --({s['relationship_type']})--> {s['target']}"
                    for s in steps
                )
                parts.append(f"- {path_info['from']} to {path_info['to']}: {path_str}")

        return "\n".join(parts) if parts else "No graph context available."

    def _format_related_papers(self, graph_context: dict) -> str:
        """Format related papers for the prompt."""
        related = graph_context.get("related_papers", [])
        if not related:
            return "No additional related papers found."

        parts = []
        seen = set()
        for rp in related:
            pid = rp.get("paper_id", "")
            if pid not in seen:
                seen.add(pid)
                shared = rp.get("shared_entities", [])[:5]
                parts.append(f"- {pid}: shares entities [{', '.join(shared)}]")

        return "\n".join(parts)

    def _collect_sources(self, search_results: list[dict]) -> list[dict]:
        """Collect unique source papers from search results."""
        sources = {}
        for result in search_results:
            pid = result.get("metadata", {}).get("paper_id")
            if pid and pid not in sources:
                sources[pid] = {
                    "paper_id": pid,
                    "page": result.get("metadata", {}).get("page"),
                    "relevance": 1 - result.get("distance", 0),
                }
        return list(sources.values())
