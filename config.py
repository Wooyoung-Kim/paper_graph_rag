"""Configuration management for Paper Graph RAG system."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env")


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    model: str = Field(default="")

    def get_model_name(self) -> str:
        if self.model:
            return self.model
        env_model = os.getenv("LLM_MODEL", "")
        if env_model:
            return env_model
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
        }
        return defaults.get(self.provider, "gpt-4o-mini")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )


class PathConfig(BaseModel):
    """Path configuration."""

    project_root: Path = _PROJECT_ROOT
    data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", _PROJECT_ROOT / "data"))
    )

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def graph_dir(self) -> Path:
        return self.data_dir / "graph"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def vault_dir(self) -> Path:
        return Path(
            os.getenv("OBSIDIAN_VAULT_DIR", str(self.data_dir / "vault"))
        )

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [
            self.papers_dir,
            self.graph_dir,
            self.chroma_dir,
            self.vault_dir / "papers",
            self.vault_dir / "entities",
        ]:
            d.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """Global settings."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Extraction parameters
    max_entities_per_chunk: int = 20
    max_relationships_per_chunk: int = 15


def get_settings() -> Settings:
    """Get global settings singleton."""
    return Settings()
