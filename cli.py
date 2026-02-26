"""CLI interface for Paper Graph RAG system."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config import get_settings

console = Console()


@click.group()
def cli():
    """📚 Paper Graph RAG — Knowledge Graph + Obsidian RAG System"""
    pass


# ============================================================================
# Ingest Command
# ============================================================================

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--batch", is_flag=True, help="Ingest all PDFs in directory")
def ingest(path: str, batch: bool):
    """Ingest PDF/Markdown paper(s) into the knowledge base."""
    from ingest import ingest_file
    from extract import extract_from_chunks, merge_extraction_results, generate_paper_summary
    from graph import KnowledgeGraph
    from vectorstore import VectorStore
    from notes import ObsidianVaultGenerator

    settings = get_settings()
    settings.paths.ensure_dirs()

    kg = KnowledgeGraph()
    vs = VectorStore()
    vault = ObsidianVaultGenerator(kg)

    target = Path(path)
    if batch or target.is_dir():
        files = sorted(list(target.glob("*.pdf")) + list(target.glob("*.md")))
        if not files:
            console.print("[red]No PDF or markdown files found in directory.[/red]")
            return
        console.print(f"[bold cyan]Found {len(files)} files to ingest.[/bold cyan]\n")
    else:
        files = [target]

    for f in files:
        file_ext = f.suffix.lower()
        console.print(Panel(f"[bold]{f.name}[/bold]", title="📄 Ingesting"))


        try:
            # Step 1: Parse file
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
                task = progress.add_task(f"Parsing {file_ext} file...", total=None)
                metadata, chunks = ingest_file(f, settings)
                progress.update(task, description=f"✅ Parsed: {len(chunks)} chunks")

            console.print(f"  Title: [green]{metadata.title}[/green]")
            if metadata.doi:
                console.print(f"  DOI: [blue]{metadata.doi}[/blue]")

            # Step 2: Extract entities
            console.print(f"\n  ⚙️  Extracting entities from {len(chunks)} chunks...")

            # Sample chunks for extraction (to reduce API calls)
            sample_size = min(len(chunks), 15)
            step = max(1, len(chunks) // sample_size)
            sampled_chunks = [chunks[i].text for i in range(0, len(chunks), step)][:sample_size]

            def progress_cb(current, total):
                console.print(f"    [{current}/{total}] chunks processed", end="\r")

            results = extract_from_chunks(sampled_chunks, progress_callback=progress_cb)
            merged = merge_extraction_results(results)

            console.print(f"\n  ✅ Extracted: {len(merged.entities)} entities, {len(merged.relationships)} relationships")

            # Step 3: Generate summary
            console.print("  ⚙️  Generating paper summary...")
            summary = generate_paper_summary(
                metadata.title,
                merged.entities,
                merged.relationships,
                metadata.abstract,
            )

            # Step 4: Update knowledge graph
            kg.add_paper_results(
                metadata.paper_id,
                merged.entities,
                merged.relationships,
            )
            kg.save()
            console.print("  ✅ Knowledge graph updated")

            # Step 5: Add to vector store
            vs.add_chunks(
                chunk_ids=[c.chunk_id for c in chunks],
                texts=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
            console.print(f"  ✅ Vector store: {vs.count} total chunks")

            # Step 6: Generate Obsidian notes
            relationships_dicts = [r.model_dump() for r in merged.relationships]
            paper_note_path = vault.write_paper_note(
                metadata, merged.entities, relationships_dicts, summary
            )
            console.print(f"  ✅ Paper note: {paper_note_path}")

            # Update entity notes
            for entity in merged.entities:
                vault.write_entity_note(entity.name)

            vault.rebuild_all_notes()
            console.print("  ✅ Entity notes updated\n")

        except Exception as e:
            console.print(f"  [red]❌ Error: {e}[/red]\n")
            import traceback
            traceback.print_exc()

    console.print(Panel("[bold green]Ingestion complete![/bold green]"))
    _print_stats(kg, vs)


# ============================================================================
# Fetch Command (PubMed)
# ============================================================================

@cli.command()
@click.argument("topic")
@click.option("--max", "-m", "max_results", default=10, help="Max papers to fetch (0=all)")
@click.option("--all", "fetch_all", is_flag=True, help="Fetch ALL search results")
@click.option("--sort", "-s", default="relevance", type=click.Choice(["relevance", "pub_date"]), help="Sort order")
@click.option("--min-date", default="", help="Min date (YYYY/MM/DD)")
@click.option("--max-date", default="", help="Max date (YYYY/MM/DD)")
def fetch(topic: str, max_results: int, fetch_all: bool, sort: str, min_date: str, max_date: str):
    """Search PubMed by topic and download papers."""
    from pubmed import fetch_papers_by_topic

    settings = get_settings()
    settings.paths.ensure_dirs()

    console.print(Panel(f"[bold]{topic}[/bold]", title="🔍 PubMed Search"))

    if fetch_all:
        max_results = 0

    results = fetch_papers_by_topic(
        topic=topic,
        max_results=max_results,
        output_dir=settings.paths.papers_dir,
        sort=sort,
        min_date=min_date,
        max_date=max_date,
    )

    if results:
        table = Table(title=f"📥 Fetched {len(results)} Papers")
        table.add_column("PMID", style="cyan")
        table.add_column("Year", style="yellow")
        table.add_column("Title", style="white", max_width=60)
        table.add_column("Type", style="green")

        for r in results:
            table.add_row(
                r["pmid"],
                r.get("year", ""),
                r["title"][:60] + ("..." if len(r["title"]) > 60 else ""),
                r.get("file_type", "?"),
            )
        console.print(table)

        console.print(f"\n📂 Papers saved to: {settings.paths.papers_dir}")
        console.print("[dim]Run [bold]./run.sh ingest data/papers/ --batch[/bold] to ingest into knowledge base[/dim]")


@cli.command(name="fetch-ingest")
@click.argument("topic")
@click.option("--max", "-m", "max_results", default=10, help="Max papers to fetch (0=all)")
@click.option("--all", "fetch_all", is_flag=True, help="Fetch ALL search results")
@click.option("--sort", "-s", default="relevance", type=click.Choice(["relevance", "pub_date"]), help="Sort order")
@click.option("--min-date", default="", help="Min date (YYYY/MM/DD)")
@click.option("--max-date", default="", help="Max date (YYYY/MM/DD)")
def fetch_ingest(topic: str, max_results: int, fetch_all: bool, sort: str, min_date: str, max_date: str):
    """Search PubMed, download papers, AND auto-ingest into knowledge base."""
    import subprocess, sys
    from pubmed import fetch_papers_by_topic

    settings = get_settings()
    settings.paths.ensure_dirs()

    console.print(Panel(f"[bold]{topic}[/bold]", title="🔍+📥 PubMed Fetch & Ingest"))

    if fetch_all:
        max_results = 0

    # Step 1: Fetch
    results = fetch_papers_by_topic(
        topic=topic,
        max_results=max_results,
        output_dir=settings.paths.papers_dir,
        sort=sort,
        min_date=min_date,
        max_date=max_date,
    )

    if not results:
        console.print("[red]No papers fetched.[/red]")
        return

    pdf_count = sum(1 for r in results if r.get("file_type") == ".pdf")
    md_count = sum(1 for r in results if r.get("file_type") == ".md")
    console.print(f"\n📄 Downloaded: {pdf_count} PDFs, {md_count} abstracts")

    # Step 2: Auto-ingest
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Starting auto-ingest...[/bold cyan]\n")

    # Call ingest for each downloaded file
    from ingest import ingest_file
    from extract import extract_from_chunks, merge_extraction_results, generate_paper_summary
    from graph import KnowledgeGraph
    from vectorstore import VectorStore
    from notes import ObsidianVaultGenerator

    kg = KnowledgeGraph()
    vs = VectorStore()
    vault = ObsidianVaultGenerator(kg)

    for r in results:
        fpath = r.get("file_path")
        if not fpath:
            continue

        fpath = Path(fpath)
        console.print(Panel(f"[bold]{fpath.name}[/bold]", title="📄 Ingesting"))

        try:
            metadata, chunks = ingest_file(fpath, settings)
            console.print(f"  Parsed: {len(chunks)} chunks")

            # Extract entities (sample)
            sample_size = min(len(chunks), 10)
            step = max(1, len(chunks) // sample_size)
            sampled = [chunks[i].text for i in range(0, len(chunks), step)][:sample_size]

            extraction_results = extract_from_chunks(sampled)
            merged = merge_extraction_results(extraction_results)
            console.print(f"  Entities: {len(merged.entities)}, Relations: {len(merged.relationships)}")

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
            for ent in merged.entities:
                vault.write_entity_note(ent.name)

            console.print(f"  ✅ Done\n")

        except Exception as e:
            console.print(f"  [red]❌ Error: {e}[/red]\n")

    vault.rebuild_all_notes()
    console.print(Panel("[bold green]Fetch & Ingest complete![/bold green]"))
    _print_stats(kg, vs)


# ============================================================================
# Query Command
# ============================================================================

@cli.command()
@click.argument("question")
@click.option("--chunks", "-n", default=8, help="Number of chunks to retrieve")
@click.option("--depth", "-d", default=1, help="Graph traversal depth")
def query(question: str, chunks: int, depth: int):
    """Query the knowledge base with RAG."""
    from rag import RAGEngine

    console.print(Panel(f"[bold]{question}[/bold]", title="❓ Query"))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Searching knowledge base...", total=None)

        engine = RAGEngine()
        result = engine.query(question, n_chunks=chunks, graph_depth=depth)

        progress.update(task, description="✅ Answer generated")

    # Display answer
    console.print(Panel(result["answer"], title="💡 Answer", border_style="green"))

    # Display sources
    if result["sources"]:
        table = Table(title="📖 Sources")
        table.add_column("Paper ID", style="cyan")
        table.add_column("Page", style="yellow")
        table.add_column("Relevance", style="green")

        for src in result["sources"]:
            table.add_row(
                src["paper_id"],
                str(src.get("page", "?")),
                f"{src.get('relevance', 0):.2f}",
            )
        console.print(table)

    # Display related entities
    if result["related_entities"]:
        entity_names = [e.get("name", "") for e in result["related_entities"][:10]]
        console.print(f"\n🔗 Related Entities: {', '.join(entity_names)}")

    # Display graph paths
    if result["graph_paths"]:
        console.print("\n🛤️  Connection Paths:")
        for path_info in result["graph_paths"]:
            steps = path_info["steps"]
            path_str = " → ".join(
                f"{s['source']} --({s['relationship_type']})--> {s['target']}"
                for s in steps
            )
            console.print(f"  {path_info['from']} ⟶ {path_info['to']}: {path_str}")


# ============================================================================
# Stats Command
# ============================================================================

@cli.command()
def stats():
    """Show knowledge base statistics."""
    from graph import KnowledgeGraph
    from vectorstore import VectorStore

    kg = KnowledgeGraph()
    vs = VectorStore()
    _print_stats(kg, vs)


def _print_stats(kg, vs):
    """Print statistics table."""
    graph_stats = kg.stats()

    table = Table(title="📊 Knowledge Base Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Entities (Nodes)", str(graph_stats["total_nodes"]))
    table.add_row("Total Relationships (Edges)", str(graph_stats["total_edges"]))
    table.add_row("Papers Indexed", str(graph_stats["total_papers"]))
    table.add_row("Vector Store Chunks", str(vs.count))
    table.add_row("Connected Components", str(graph_stats["connected_components"]))

    console.print(table)

    if graph_stats["entity_types"]:
        type_table = Table(title="Entity Types")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green")
        for etype, count in sorted(graph_stats["entity_types"].items(), key=lambda x: -x[1]):
            type_table.add_row(etype, str(count))
        console.print(type_table)


# ============================================================================
# Rebuild Notes Command
# ============================================================================

@cli.command(name="rebuild-notes")
def rebuild_notes():
    """Rebuild all Obsidian vault notes from the knowledge graph."""
    from graph import KnowledgeGraph
    from notes import ObsidianVaultGenerator

    console.print("[bold cyan]Rebuilding Obsidian vault notes...[/bold cyan]")

    kg = KnowledgeGraph()
    vault = ObsidianVaultGenerator(kg)
    result = vault.rebuild_all_notes()

    console.print(f"✅ Rebuilt {result['entity_notes']} entity notes")
    console.print(f"✅ Updated vault index")


# ============================================================================
# Search Entities Command
# ============================================================================

@cli.command(name="search-entity")
@click.argument("name")
@click.option("--depth", "-d", default=1, help="Graph traversal depth")
def search_entity(name: str, depth: int):
    """Search for an entity in the knowledge graph."""
    from graph import KnowledgeGraph

    kg = KnowledgeGraph()
    entity = kg.get_entity(name)

    if entity is None:
        # Try case-insensitive search
        for node in kg.graph.nodes():
            if node.lower() == name.lower():
                entity = kg.get_entity(node)
                name = node
                break

    if entity is None:
        console.print(f"[red]Entity '{name}' not found.[/red]")
        # Suggest similar
        similar = [n for n in kg.graph.nodes() if name.lower() in n.lower()]
        if similar:
            console.print(f"Did you mean: {', '.join(similar[:5])}?")
        return

    console.print(Panel(f"[bold]{name}[/bold] ({entity.get('type', 'Unknown')})", title="🔬 Entity"))

    if entity.get("aliases"):
        console.print(f"Aliases: {', '.join(entity['aliases'])}")
    if entity.get("description"):
        console.print(f"Description: {entity['description']}")

    papers = entity.get("papers", [])
    if papers:
        console.print(f"\n📄 Mentioned in {len(papers)} paper(s): {', '.join(papers)}")

    neighbors = kg.get_neighbors(name, depth=depth)
    if neighbors["edges"]:
        console.print(f"\n🔗 Relationships ({len(neighbors['edges'])}):")
        for edge in neighbors["edges"]:
            console.print(f"  {edge['source']} → {edge['relationship_type']} → {edge['target']}")


if __name__ == "__main__":
    cli()
