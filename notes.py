"""Obsidian-compatible markdown note generator."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_settings
from extract import Entity, EntityType
from graph import KnowledgeGraph
from ingest import PaperMetadata


def _sanitize_filename(name: str) -> str:
    """Convert entity/paper name to a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('_')[:100]


def _wikilink(name: str) -> str:
    """Create an Obsidian wikilink."""
    return f"[[{name}]]"


# ============================================================================
# Paper Note
# ============================================================================

def generate_paper_note(
    metadata: PaperMetadata,
    entities: list[Entity],
    relationships: list[dict[str, Any]],
    summary: str = "",
    related_papers: list[dict[str, Any]] | None = None,
) -> str:
    """Generate an Obsidian-compatible markdown note for a paper."""

    # YAML frontmatter
    tags = ["paper"]
    entity_types_found = {e.type.value.lower() for e in entities}
    tags.extend(sorted(entity_types_found))

    authors_str = json.dumps(metadata.authors, ensure_ascii=False) if metadata.authors else "[]"

    note = f"""---
title: "{metadata.title}"
authors: {authors_str}
doi: "{metadata.doi}"
paper_id: "{metadata.paper_id}"
filename: "{metadata.filename}"
date_added: "{datetime.now().strftime('%Y-%m-%d')}"
tags: [{', '.join(tags)}]
---

# {metadata.title}

"""

    if metadata.authors:
        note += f"**Authors**: {', '.join(metadata.authors)}\n"
    if metadata.doi:
        note += f"**DOI**: [{metadata.doi}](https://doi.org/{metadata.doi})\n"
    note += f"**Pages**: {metadata.total_pages}\n\n"

    # Summary
    if summary:
        note += f"## Summary\n\n{summary}\n\n"
    elif metadata.abstract:
        note += f"## Abstract\n\n{metadata.abstract}\n\n"

    # Key Entities by type
    note += "## Key Entities\n\n"
    entities_by_type: dict[str, list[str]] = {}
    for entity in entities:
        type_label = entity.type.value
        if type_label not in entities_by_type:
            entities_by_type[type_label] = []
        entities_by_type[type_label].append(entity.name)

    for type_label in sorted(entities_by_type.keys()):
        names = entities_by_type[type_label]
        links = ", ".join(_wikilink(n) for n in sorted(set(names)))
        note += f"- **{type_label}**: {links}\n"

    note += "\n"

    # Key Relationships
    if relationships:
        note += "## Key Relationships\n\n"
        for rel in relationships:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            rel_type = rel.get("relationship_type", "related_to")
            note += f"- {_wikilink(src)} → *{rel_type}* → {_wikilink(tgt)}\n"
        note += "\n"

    # Related Papers
    if related_papers:
        note += "## Related Papers\n\n"
        for rp in related_papers:
            pid = rp.get("paper_id", "")
            shared = rp.get("shared_entities", [])
            shared_links = ", ".join(_wikilink(e) for e in shared[:5])
            more = f" (+{len(shared)-5} more)" if len(shared) > 5 else ""
            note += f"- {_wikilink(pid)} — shared: {shared_links}{more}\n"
        note += "\n"

    return note


# ============================================================================
# Entity Note
# ============================================================================

def generate_entity_note(
    entity_name: str,
    entity_data: dict[str, Any],
    graph: KnowledgeGraph,
    paper_metadata: dict[str, PaperMetadata] | None = None,
) -> str:
    """Generate an Obsidian-compatible markdown note for an entity."""

    entity_type = entity_data.get("type", "Unknown")
    aliases = entity_data.get("aliases", [])
    description = entity_data.get("description", "")
    papers = entity_data.get("papers", [])

    tags = ["entity", entity_type.lower()]

    note = f"""---
entity_name: "{entity_name}"
entity_type: "{entity_type}"
aliases: {json.dumps(aliases, ensure_ascii=False)}
tags: [{', '.join(tags)}]
---

# {entity_name}

**Type**: {entity_type}
"""

    if aliases:
        note += f"**Aliases**: {', '.join(aliases)}\n"
    if description:
        note += f"\n{description}\n"

    note += "\n"

    # Relationships from graph
    neighbors = graph.get_neighbors(entity_name, depth=1)
    outgoing = [e for e in neighbors["edges"] if e["source"] == entity_name]
    incoming = [e for e in neighbors["edges"] if e["target"] == entity_name]

    if outgoing:
        note += "## Outgoing Relationships\n\n"
        for edge in outgoing:
            note += f"- → *{edge['relationship_type']}* → {_wikilink(edge['target'])}\n"
        note += "\n"

    if incoming:
        note += "## Incoming Relationships\n\n"
        for edge in incoming:
            note += f"- {_wikilink(edge['source'])} → *{edge['relationship_type']}* →\n"
        note += "\n"

    # Papers mentioning this entity
    if papers:
        note += "## Mentioned In\n\n"
        for pid in papers:
            if paper_metadata and pid in paper_metadata:
                title = paper_metadata[pid].title
                note += f"- {_wikilink(pid)} — {title}\n"
            else:
                note += f"- {_wikilink(pid)}\n"
        note += "\n"

    return note


# ============================================================================
# Vault Generator
# ============================================================================

import json


class ObsidianVaultGenerator:
    """Generate and maintain an Obsidian vault from the knowledge graph."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        vault_dir: str | Path | None = None,
    ):
        self.graph = graph
        if vault_dir is None:
            settings = get_settings()
            settings.paths.ensure_dirs()
            self.vault_dir = settings.paths.vault_dir
        else:
            self.vault_dir = Path(vault_dir)

        self.papers_dir = self.vault_dir / "papers"
        self.entities_dir = self.vault_dir / "entities"
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.entities_dir.mkdir(parents=True, exist_ok=True)

        # Store for paper metadata
        self._paper_metadata: dict[str, PaperMetadata] = {}

    def register_paper(self, metadata: PaperMetadata) -> None:
        """Register paper metadata for note generation."""
        self._paper_metadata[metadata.paper_id] = metadata

    def write_paper_note(
        self,
        metadata: PaperMetadata,
        entities: list[Entity],
        relationships: list[dict[str, Any]],
        summary: str = "",
    ) -> Path:
        """Write a paper note to the vault."""
        self.register_paper(metadata)

        related = self.graph.find_related_papers(metadata.paper_id, min_shared=1)
        note_content = generate_paper_note(
            metadata, entities, relationships, summary, related
        )

        filename = _sanitize_filename(metadata.paper_id) + ".md"
        filepath = self.papers_dir / filename
        filepath.write_text(note_content, encoding="utf-8")
        return filepath

    def write_entity_note(self, entity_name: str) -> Path | None:
        """Write an entity note to the vault."""
        entity_data = self.graph.get_entity(entity_name)
        if entity_data is None:
            return None

        note_content = generate_entity_note(
            entity_name, entity_data, self.graph, self._paper_metadata
        )

        filename = _sanitize_filename(entity_name) + ".md"
        filepath = self.entities_dir / filename
        filepath.write_text(note_content, encoding="utf-8")
        return filepath

    def rebuild_all_notes(self) -> dict[str, int]:
        """Rebuild all entity notes from the current graph state."""
        entity_count = 0
        for name, data in self.graph.graph.nodes(data=True):
            self.write_entity_note(name)
            entity_count += 1

        # Write index
        self._write_index()

        return {"entity_notes": entity_count, "paper_notes": len(self._paper_metadata)}

    def _write_index(self) -> None:
        """Write a vault index note."""
        stats = self.graph.stats()

        index = f"""---
tags: [index]
date_updated: "{datetime.now().strftime('%Y-%m-%d %H:%M')}"
---

# 📚 Paper Graph RAG — Knowledge Vault

## Statistics

| Metric | Count |
|--------|-------|
| Total Entities | {stats['total_nodes']} |
| Total Relationships | {stats['total_edges']} |
| Total Papers | {stats['total_papers']} |
| Connected Components | {stats['connected_components']} |

## Entity Types

"""
        for etype, count in sorted(stats.get("entity_types", {}).items()):
            index += f"- **{etype}**: {count}\n"

        index += "\n## Recent Papers\n\n"
        for pid, meta in list(self._paper_metadata.items())[-10:]:
            index += f"- {_wikilink(pid)} — {meta.title}\n"

        index += "\n## Relationship Types\n\n"
        for rtype, count in sorted(stats.get("relationship_types", {}).items()):
            index += f"- *{rtype}*: {count}\n"

        index_path = self.vault_dir / "_index.md"
        index_path.write_text(index, encoding="utf-8")
