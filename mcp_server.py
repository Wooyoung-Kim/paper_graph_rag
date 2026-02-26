"""MCP (Model Context Protocol) server for Paper Graph RAG.

Exposes the knowledge base as tools that LLMs (Claude, etc.) can use directly.
Run with: python mcp_server.py
Or register in Claude Desktop / .claude.json MCP config.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

# Fix libstdc++ for ChromaDB
_conda_base = os.environ.get("CONDA_PREFIX", os.path.expanduser("~/miniconda3/envs/paper_rag"))
os.environ["LD_LIBRARY_PATH"] = f"{_conda_base}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

from fastmcp import FastMCP

mcp = FastMCP(
    name="paper-graph-rag",
    instructions=(
        "A biomedical research paper knowledge base with knowledge graph and RAG capabilities. "
        "Use these tools to search PubMed, query indexed papers, explore entity relationships, "
        "and find connections between biomedical concepts like genes, pathways, diseases, and cell types."
    ),
)


# ============================================================================
# Tool: Search PubMed
# ============================================================================

@mcp.tool()
def search_pubmed_papers(
    topic: str,
    max_results: int = 10,
    sort: str = "relevance",
    min_date: str = "",
    max_date: str = "",
) -> dict[str, Any]:
    """Search PubMed for papers on a given topic.

    Args:
        topic: Search query (e.g., "scRNA-seq germinal center B cell")
        max_results: Max papers to return (0 for all results)
        sort: Sort order - "relevance" or "pub_date"
        min_date: Min publication date (YYYY/MM/DD)
        max_date: Max publication date (YYYY/MM/DD)

    Returns:
        Dict with list of papers (pmid, title, authors, abstract, doi, journal, year)
    """
    from pubmed import search_pubmed, fetch_articles

    pmids = search_pubmed(
        topic, max_results=max_results, sort=sort,
        min_date=min_date, max_date=max_date,
    )

    if not pmids:
        return {"count": 0, "papers": []}

    articles = fetch_articles(pmids)

    papers = []
    for a in articles:
        papers.append({
            "pmid": a.pmid,
            "title": a.title,
            "authors": a.authors[:5],
            "abstract": a.abstract[:500] + ("..." if len(a.abstract) > 500 else ""),
            "doi": a.doi,
            "journal": a.journal,
            "year": a.year,
            "keywords": a.keywords,
            "has_fulltext": a.has_fulltext,
        })

    return {"count": len(papers), "papers": papers}


# ============================================================================
# Tool: Fetch & Ingest Papers
# ============================================================================

@mcp.tool()
def fetch_and_ingest_papers(
    topic: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search PubMed, download papers, and ingest them into the knowledge base.

    This downloads PDFs (or abstracts) and runs entity extraction + knowledge graph
    construction. Use sparingly as it calls an LLM for each paper.

    Args:
        topic: Search query
        max_results: Max papers to fetch and ingest (0 for all)

    Returns:
        Dict with ingestion results (papers processed, entities found, etc.)
    """
    from pubmed import fetch_papers_by_topic
    from ingest import ingest_file
    from extract import extract_from_chunks, merge_extraction_results, generate_paper_summary
    from graph import KnowledgeGraph
    from vectorstore import VectorStore
    from notes import ObsidianVaultGenerator
    from config import get_settings

    settings = get_settings()
    settings.paths.ensure_dirs()

    # Download papers
    results = fetch_papers_by_topic(
        topic=topic,
        max_results=max_results,
        output_dir=settings.paths.papers_dir,
    )

    if not results:
        return {"status": "no_papers_found", "count": 0}

    kg = KnowledgeGraph()
    vs = VectorStore()
    vault = ObsidianVaultGenerator(kg)

    ingested = []
    for r in results:
        fpath = r.get("file_path")
        if not fpath:
            continue

        try:
            metadata, chunks = ingest_file(Path(fpath), settings)

            sample_size = min(len(chunks), 10)
            step = max(1, len(chunks) // sample_size)
            sampled = [chunks[i].text for i in range(0, len(chunks), step)][:sample_size]

            extraction_results = extract_from_chunks(sampled)
            merged = merge_extraction_results(extraction_results)

            summary = generate_paper_summary(
                metadata.title, merged.entities, merged.relationships, metadata.abstract
            )

            kg.add_paper_results(metadata.paper_id, merged.entities, merged.relationships)
            kg.save()

            vs.add_chunks(
                [c.chunk_id for c in chunks],
                [c.text for c in chunks],
                [c.metadata for c in chunks],
            )

            rels = [rel.model_dump() for rel in merged.relationships]
            vault.write_paper_note(metadata, merged.entities, rels, summary)

            ingested.append({
                "paper_id": metadata.paper_id,
                "title": metadata.title,
                "entities": len(merged.entities),
                "relationships": len(merged.relationships),
            })
        except Exception as e:
            ingested.append({"file": fpath, "error": str(e)})

    vault.rebuild_all_notes()

    return {
        "status": "success",
        "papers_ingested": len([i for i in ingested if "error" not in i]),
        "details": ingested,
    }


# ============================================================================
# Tool: Query Knowledge Base (RAG)
# ============================================================================

@mcp.tool()
def query_knowledge_base(
    question: str,
    n_chunks: int = 8,
    graph_depth: int = 1,
) -> dict[str, Any]:
    """Query the indexed paper knowledge base using RAG with graph enhancement.

    Searches the vector store for relevant chunks, expands context via the
    knowledge graph, and generates an answer with citations.

    Args:
        question: Your question about the indexed papers
        n_chunks: Number of text chunks to retrieve (default 8)
        graph_depth: Depth of knowledge graph traversal (default 1)

    Returns:
        Dict with answer, source papers, related entities, and graph paths
    """
    from rag import RAGEngine

    engine = RAGEngine()
    result = engine.query(question, n_chunks=n_chunks, graph_depth=graph_depth)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "related_entities": [
            {"name": e.get("name"), "type": e.get("type")}
            for e in result.get("related_entities", [])[:10]
        ],
        "graph_paths": result.get("graph_paths", []),
    }


# ============================================================================
# Tool: Search Entity
# ============================================================================

@mcp.tool()
def search_entity(
    name: str,
    depth: int = 1,
) -> dict[str, Any]:
    """Search for a biomedical entity in the knowledge graph.

    Returns entity details, relationships, and connected papers.

    Args:
        name: Entity name (e.g., "CD19", "TP53", "Germinal Center B cell")
        depth: Graph traversal depth (1 or 2)

    Returns:
        Dict with entity info, relationships, and papers
    """
    from graph import KnowledgeGraph

    kg = KnowledgeGraph()

    # Try exact match first, then case-insensitive
    entity = kg.get_entity(name)
    if entity is None:
        for node in kg.graph.nodes():
            if node.lower() == name.lower():
                entity = kg.get_entity(node)
                name = node
                break

    if entity is None:
        # Suggest similar entities
        similar = [n for n in kg.graph.nodes() if name.lower() in n.lower()]
        return {"found": False, "suggestions": similar[:10]}

    neighbors = kg.get_neighbors(name, depth=depth)

    return {
        "found": True,
        "name": name,
        "type": entity.get("type"),
        "aliases": entity.get("aliases", []),
        "description": entity.get("description", ""),
        "papers": entity.get("papers", []),
        "relationships": [
            {
                "source": e["source"],
                "target": e["target"],
                "type": e["relationship_type"],
            }
            for e in neighbors.get("edges", [])
        ],
    }


# ============================================================================
# Tool: Get Knowledge Base Stats
# ============================================================================

@mcp.tool()
def get_knowledge_base_stats() -> dict[str, Any]:
    """Get statistics about the current knowledge base.

    Returns counts of entities, relationships, papers, and entity type breakdown.
    """
    from graph import KnowledgeGraph
    from vectorstore import VectorStore

    kg = KnowledgeGraph()
    vs = VectorStore()
    stats = kg.stats()

    return {
        "total_entities": stats["total_nodes"],
        "total_relationships": stats["total_edges"],
        "total_papers": stats["total_papers"],
        "vector_chunks": vs.count,
        "entity_types": stats.get("entity_types", {}),
        "relationship_types": stats.get("relationship_types", {}),
    }


# ============================================================================
# Tool: Find Related Papers
# ============================================================================

@mcp.tool()
def find_related_papers(
    paper_id: str,
    min_shared_entities: int = 2,
) -> dict[str, Any]:
    """Find papers related to a given paper by shared entities.

    Args:
        paper_id: Paper ID (e.g., "paper_abc123")
        min_shared_entities: Minimum shared entities to consider related

    Returns:
        Dict with list of related papers and their shared entities
    """
    from graph import KnowledgeGraph

    kg = KnowledgeGraph()
    related = kg.find_related_papers(paper_id, min_shared=min_shared_entities)

    return {
        "paper_id": paper_id,
        "related_papers": related[:20],
    }


# ============================================================================
# Tool: Find Entity Connection Path
# ============================================================================

@mcp.tool()
def find_entity_connection(
    entity_a: str,
    entity_b: str,
) -> dict[str, Any]:
    """Find how two biomedical entities are connected in the knowledge graph.

    Args:
        entity_a: First entity name (e.g., "TP53")
        entity_b: Second entity name (e.g., "Lymphoma")

    Returns:
        Connection path with relationship types, or None if not connected
    """
    from graph import KnowledgeGraph

    kg = KnowledgeGraph()
    path = kg.find_path(entity_a, entity_b)

    if path is None:
        return {
            "connected": False,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "message": "No connection path found between these entities.",
        }

    return {
        "connected": True,
        "entity_a": entity_a,
        "entity_b": entity_b,
        "path": path,
        "path_length": len(path),
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    mcp.run()
