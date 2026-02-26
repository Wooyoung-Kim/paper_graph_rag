"""Knowledge Graph management using NetworkX with JSON persistence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from config import get_settings
from extract import Entity, EntityType, Relationship


class KnowledgeGraph:
    """
    Knowledge graph for biomedical entities and relationships.
    Backed by NetworkX DiGraph with JSON serialization.
    """

    def __init__(self, graph_path: str | Path | None = None):
        self.graph = nx.DiGraph()
        self._paper_entities: dict[str, list[str]] = defaultdict(list)  # paper_id -> entity names

        if graph_path is None:
            settings = get_settings()
            settings.paths.ensure_dirs()
            self.graph_path = settings.paths.graph_dir / "knowledge_graph.json"
        else:
            self.graph_path = Path(graph_path)

        if self.graph_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Add / Update
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity, paper_id: str) -> None:
        """Add or update an entity node."""
        name = entity.name
        if self.graph.has_node(name):
            # Update existing node
            node = self.graph.nodes[name]
            papers = set(node.get("papers", []))
            papers.add(paper_id)
            node["papers"] = sorted(papers)

            aliases = set(node.get("aliases", []))
            aliases.update(entity.aliases)
            node["aliases"] = sorted(aliases)

            if entity.description and not node.get("description"):
                node["description"] = entity.description
        else:
            self.graph.add_node(
                name,
                type=entity.type.value,
                aliases=sorted(set(entity.aliases)),
                description=entity.description,
                papers=[paper_id],
            )

        # Track paper -> entity mapping
        if name not in self._paper_entities[paper_id]:
            self._paper_entities[paper_id].append(name)

    def add_relationship(self, rel: Relationship, paper_id: str) -> None:
        """Add or update a relationship edge."""
        src, tgt = rel.source, rel.target

        # Ensure both nodes exist
        for node_name in [src, tgt]:
            if not self.graph.has_node(node_name):
                self.graph.add_node(
                    node_name,
                    type="Unknown",
                    aliases=[],
                    description="",
                    papers=[paper_id],
                )

        edge_key = (src, tgt)
        if self.graph.has_edge(*edge_key):
            edge = self.graph.edges[edge_key]
            papers = set(edge.get("papers", []))
            papers.add(paper_id)
            edge["papers"] = sorted(papers)

            # Append evidence
            evidences = edge.get("evidences", [])
            if rel.evidence and rel.evidence not in evidences:
                evidences.append(rel.evidence)
            edge["evidences"] = evidences
        else:
            self.graph.add_edge(
                src,
                tgt,
                relationship_type=rel.relationship_type,
                evidences=[rel.evidence] if rel.evidence else [],
                papers=[paper_id],
            )

    def add_paper_results(
        self,
        paper_id: str,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Add all entities and relationships from a paper."""
        for entity in entities:
            self.add_entity(entity, paper_id)
        for rel in relationships:
            self.add_relationship(rel, paper_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Get entity node data."""
        if self.graph.has_node(name):
            return {"name": name, **dict(self.graph.nodes[name])}
        return None

    def get_neighbors(self, entity_name: str, depth: int = 1) -> dict[str, Any]:
        """Get neighboring entities and relationships."""
        if not self.graph.has_node(entity_name):
            return {"entity": entity_name, "neighbors": [], "edges": []}

        # BFS to specified depth
        visited = {entity_name}
        frontier = {entity_name}
        all_neighbors = []
        all_edges = []

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                # Outgoing edges
                for _, target, data in self.graph.out_edges(node, data=True):
                    all_edges.append({
                        "source": node,
                        "target": target,
                        "relationship_type": data.get("relationship_type", "related_to"),
                        "papers": data.get("papers", []),
                    })
                    if target not in visited:
                        next_frontier.add(target)
                        visited.add(target)

                # Incoming edges
                for source, _, data in self.graph.in_edges(node, data=True):
                    all_edges.append({
                        "source": source,
                        "target": node,
                        "relationship_type": data.get("relationship_type", "related_to"),
                        "papers": data.get("papers", []),
                    })
                    if source not in visited:
                        next_frontier.add(source)
                        visited.add(source)

            all_neighbors.extend(
                {"name": n, **dict(self.graph.nodes[n])} for n in next_frontier if self.graph.has_node(n)
            )
            frontier = next_frontier

        return {
            "entity": entity_name,
            "neighbors": all_neighbors,
            "edges": all_edges,
        }

    def find_path(self, source: str, target: str) -> list[dict] | None:
        """Find shortest path between two entities with relationship details."""
        if not (self.graph.has_node(source) and self.graph.has_node(target)):
            return None

        try:
            # Try directed path first
            path = nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            try:
                # Try undirected path
                undirected = self.graph.to_undirected()
                path = nx.shortest_path(undirected, source, target)
            except nx.NetworkXNoPath:
                return None

        result = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            edge_data = self.graph.get_edge_data(s, t) or self.graph.get_edge_data(t, s) or {}
            result.append({
                "source": s,
                "target": t,
                "relationship_type": edge_data.get("relationship_type", "related_to"),
                "evidence": edge_data.get("evidences", []),
            })

        return result

    def get_papers_for_entity(self, entity_name: str) -> list[str]:
        """Get all paper IDs that mention an entity."""
        if self.graph.has_node(entity_name):
            return self.graph.nodes[entity_name].get("papers", [])
        return []

    def get_entities_for_paper(self, paper_id: str) -> list[dict]:
        """Get all entities from a specific paper."""
        entities = []
        for name in self._paper_entities.get(paper_id, []):
            if self.graph.has_node(name):
                entities.append({"name": name, **dict(self.graph.nodes[name])})
        return entities

    def get_shared_entities(self, paper_id_1: str, paper_id_2: str) -> list[str]:
        """Find entities shared between two papers."""
        entities_1 = set(self._paper_entities.get(paper_id_1, []))
        entities_2 = set(self._paper_entities.get(paper_id_2, []))
        return sorted(entities_1 & entities_2)

    def find_related_papers(self, paper_id: str, min_shared: int = 2) -> list[dict]:
        """Find papers related to a given paper by shared entities."""
        my_entities = set(self._paper_entities.get(paper_id, []))
        related = defaultdict(set)

        for entity_name in my_entities:
            if self.graph.has_node(entity_name):
                for pid in self.graph.nodes[entity_name].get("papers", []):
                    if pid != paper_id:
                        related[pid].add(entity_name)

        results = []
        for pid, shared in related.items():
            if len(shared) >= min_shared:
                results.append({
                    "paper_id": pid,
                    "shared_entities": sorted(shared),
                    "shared_count": len(shared),
                })

        return sorted(results, key=lambda x: x["shared_count"], reverse=True)

    def get_entities_by_type(self, entity_type: str) -> list[dict]:
        """Get all entities of a specific type."""
        entities = []
        for name, data in self.graph.nodes(data=True):
            if data.get("type") == entity_type:
                entities.append({"name": name, **data})
        return entities

    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        type_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            type_counts[data.get("type", "Unknown")] += 1

        rel_type_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel_type_counts[data.get("relationship_type", "unknown")] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": dict(type_counts),
            "relationship_types": dict(rel_type_counts),
            "total_papers": len(self._paper_entities),
            "connected_components": nx.number_weakly_connected_components(self.graph),
        }

    # ------------------------------------------------------------------
    # Persistence (JSON)
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save graph to JSON file."""
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": [],
            "edges": [],
            "paper_entities": dict(self._paper_entities),
        }

        for name, attrs in self.graph.nodes(data=True):
            data["nodes"].append({"name": name, **attrs})

        for src, tgt, attrs in self.graph.edges(data=True):
            data["edges"].append({"source": src, "target": tgt, **attrs})

        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """Load graph from JSON file."""
        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        self._paper_entities.clear()

        for node_data in data.get("nodes", []):
            name = node_data.pop("name")
            self.graph.add_node(name, **node_data)

        for edge_data in data.get("edges", []):
            src = edge_data.pop("source")
            tgt = edge_data.pop("target")
            self.graph.add_edge(src, tgt, **edge_data)

        for paper_id, entities in data.get("paper_entities", {}).items():
            self._paper_entities[paper_id] = entities
