"""Biomedical entity and relationship extraction using LLMs."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from config import LLMConfig, get_settings


# ============================================================================
# Entity & Relationship Models
# ============================================================================

class EntityType(str, Enum):
    GENE = "Gene"
    PROTEIN = "Protein"
    CELL_TYPE = "CellType"
    DISEASE = "Disease"
    PATHWAY = "Pathway"
    DRUG = "Drug"
    EXPERIMENTAL_CONDITION = "ExperimentalCondition"
    METHOD = "Method"
    ORGANISM = "Organism"
    TISSUE = "Tissue"


class Entity(BaseModel):
    """A biomedical entity extracted from text."""

    name: str = Field(description="Canonical name of the entity")
    type: EntityType = Field(description="Type of the entity")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    description: str = Field(default="", description="Brief description from context")


class Relationship(BaseModel):
    """A relationship between two entities."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Type of relationship (e.g., regulates, inhibits, expressed_in)")
    evidence: str = Field(default="", description="Supporting text from the paper")


class ExtractionResult(BaseModel):
    """Result of entity and relationship extraction from a chunk."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief summary of the chunk content")


# ============================================================================
# Extraction Prompt
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a biomedical knowledge extraction expert.
Extract entities and relationships from the given text chunk of a research paper.

## Entity Types to Extract:
- Gene: Gene symbols and names (e.g., TP53, BRCA1, CD19)
- Protein: Protein names (e.g., p53, PD-L1)
- CellType: Cell types (e.g., T cell, B cell, macrophage, Germinal Center B cell)
- Disease: Diseases and conditions (e.g., lymphoma, COVID-19)
- Pathway: Signaling pathways (e.g., NF-κB pathway, JAK-STAT pathway)
- Drug: Drugs and compounds (e.g., dexamethasone, anti-PD1)
- ExperimentalCondition: Experimental conditions (e.g., LPS stimulation, 37°C incubation)
- Method: Methods and techniques (e.g., scRNA-seq, flow cytometry, CRISPR)
- Organism: Model organisms (e.g., mouse, human, C. elegans)
- Tissue: Tissues and organs (e.g., spleen, bone marrow, lymph node)

## Relationship Types:
- regulates, upregulates, downregulates, activates, inhibits
- expressed_in, located_in, part_of
- treats, causes, associated_with
- interacts_with, binds_to
- marker_of, used_for

## Rules:
1. Use canonical/standard names when possible (e.g., "CD19" not "cd19")
2. Include aliases if mentioned (e.g., name: "TP53", aliases: ["p53", "tumor protein p53"])
3. Only extract entities and relationships that are explicitly mentioned
4. Provide evidence text for relationships (the sentence where it was mentioned)
5. Be precise - avoid generic terms unless they are specifically discussed

Respond in JSON format matching the schema."""

EXTRACTION_USER_PROMPT = """Extract biomedical entities and relationships from this text:

---
{text}
---

Return a JSON with this structure:
{{
  "entities": [
    {{"name": "...", "type": "Gene|Protein|CellType|Disease|Pathway|Drug|ExperimentalCondition|Method|Organism|Tissue", "aliases": [...], "description": "..."}}
  ],
  "relationships": [
    {{"source": "entity_name", "target": "entity_name", "relationship_type": "...", "evidence": "..."}}
  ],
  "summary": "Brief 1-2 sentence summary of the text content"
}}"""

PAPER_SUMMARY_PROMPT = """Based on the following extracted information from a research paper, write a concise summary (3-5 sentences) covering:
1. Main research question/objective
2. Key methods used
3. Main findings
4. Significance/implications

Paper title: {title}
Extracted entities: {entities}
Extracted relationships: {relationships}
Abstract (if available): {abstract}
"""


# ============================================================================
# LLM Client
# ============================================================================

def _get_llm(config: LLMConfig | None = None):
    """Get LLM instance based on configuration."""
    if config is None:
        config = get_settings().llm

    model_name = config.get_model_name()

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _parse_extraction_response(response_text: str) -> ExtractionResult:
    """Parse LLM JSON response into ExtractionResult."""
    # Try to extract JSON from potential markdown code blocks
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code block markers)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return ExtractionResult(**data)
    except (json.JSONDecodeError, Exception) as e:
        # Return empty result on parse failure
        print(f"  [WARN] Failed to parse extraction response: {e}")
        return ExtractionResult()


# ============================================================================
# Main Extraction Functions
# ============================================================================

def extract_from_chunk(text: str, llm_config: LLMConfig | None = None) -> ExtractionResult:
    """Extract entities and relationships from a single text chunk."""
    llm = _get_llm(llm_config)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": EXTRACTION_USER_PROMPT.format(text=text)},
    ]

    response = llm.invoke(messages)
    return _parse_extraction_response(response.content)


def extract_from_chunks(
    chunks: list[dict[str, Any]],
    llm_config: LLMConfig | None = None,
    progress_callback: Any | None = None,
) -> list[ExtractionResult]:
    """
    Extract entities and relationships from multiple chunks.

    Args:
        chunks: List of dicts with at least 'text' key
        llm_config: LLM configuration override
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        List of ExtractionResult, one per chunk
    """
    results = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        if not text.strip():
            results.append(ExtractionResult())
            continue

        try:
            result = extract_from_chunk(text, llm_config)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Extraction failed for chunk {i}: {e}")
            results.append(ExtractionResult())

        if progress_callback:
            progress_callback(i + 1, total)

    return results


def generate_paper_summary(
    title: str,
    entities: list[Entity],
    relationships: list[Relationship],
    abstract: str = "",
    llm_config: LLMConfig | None = None,
) -> str:
    """Generate a summary for a paper based on extracted information."""
    llm = _get_llm(llm_config)

    entity_strs = [f"{e.name} ({e.type.value})" for e in entities[:30]]
    rel_strs = [f"{r.source} → {r.relationship_type} → {r.target}" for r in relationships[:20]]

    prompt = PAPER_SUMMARY_PROMPT.format(
        title=title,
        entities=", ".join(entity_strs) if entity_strs else "None extracted",
        relationships="\n".join(rel_strs) if rel_strs else "None extracted",
        abstract=abstract or "Not available",
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()


def merge_extraction_results(results: list[ExtractionResult]) -> ExtractionResult:
    """Merge multiple extraction results, deduplicating entities."""
    all_entities: dict[str, Entity] = {}
    all_relationships: list[Relationship] = []
    summaries: list[str] = []

    for result in results:
        for entity in result.entities:
            key = entity.name.lower()
            if key in all_entities:
                # Merge aliases
                existing = all_entities[key]
                new_aliases = set(existing.aliases) | set(entity.aliases)
                existing.aliases = list(new_aliases)
                if not existing.description and entity.description:
                    existing.description = entity.description
            else:
                all_entities[key] = entity

        all_relationships.extend(result.relationships)

        if result.summary:
            summaries.append(result.summary)

    # Deduplicate relationships
    seen_rels = set()
    unique_rels = []
    for rel in all_relationships:
        key = (rel.source.lower(), rel.target.lower(), rel.relationship_type.lower())
        if key not in seen_rels:
            seen_rels.add(key)
            unique_rels.append(rel)

    return ExtractionResult(
        entities=list(all_entities.values()),
        relationships=unique_rels,
        summary=" | ".join(summaries[:3]) if summaries else "",
    )
