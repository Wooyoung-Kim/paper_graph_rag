#!/bin/bash
# Paper Graph RAG — run script
# Usage: ./run.sh ingest /path/to/paper.pdf
#        ./run.sh query "your question here"
#        ./run.sh stats
#        ./run.sh rebuild-notes
#        ./run.sh search-entity "TP53"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="paper_rag"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"

# Fix libstdc++ for ChromaDB
export LD_LIBRARY_PATH="$CONDA_BASE/envs/$CONDA_ENV/lib:$LD_LIBRARY_PATH"

# Run CLI
conda run --no-banner -n "$CONDA_ENV" python "$SCRIPT_DIR/cli.py" "$@"
