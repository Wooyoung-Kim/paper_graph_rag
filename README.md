# 📚 Paper Graph RAG

논문 PDF → 지식 그래프 + Obsidian 노트 + RAG 시스템

## Quick Start

```bash
# 1. .env 설정 (API 키 필수)
cp .env.template .env
# .env 파일에서 OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 설정

# 2. PubMed에서 주제별 논문 가져오기
./run.sh fetch "scRNA-seq germinal center B cell" --max 10

# 3. PubMed 검색 + 자동 인제스트 (한 번에)
./run.sh fetch-ingest "vaccine immune response single cell" --max 5

# 4. 로컬 PDF 인제스트
./run.sh ingest /path/to/paper.pdf          # 단일 PDF
./run.sh ingest /path/to/papers/ --batch    # 폴더 전체

# 5. RAG 쿼리
./run.sh query "BCL6와 germinal center B cell의 관계는?"

# 6. 통계 확인
./run.sh stats

# 7. 개체 검색
./run.sh search-entity "CD19" --depth 2

# 8. Obsidian vault 열기
# data/vault/ 을 Obsidian에서 Open Vault로 열면 Graph View에서 시각화 가능
```

## Architecture

```
PubMed Topic Search → Fetch Metadata + Download PDF/Abstract
        ↓
PDF/MD → Parse (PyMuPDF) → Chunk → LLM Entity Extraction → Knowledge Graph (NetworkX)
                                                           → Vector Store (ChromaDB)
                                                           → Obsidian Notes (Markdown)
Query → Vector Search + Graph Expansion → LLM Answer + Sources + Linked Concepts
```

## Project Structure

| File | Description |
|------|-------------|
| `cli.py` | CLI (fetch, fetch-ingest, ingest, query, stats, rebuild-notes, search-entity) |
| `pubmed.py` | PubMed 주제 검색 + 논문 다운로드 (PMC/Unpaywall) |
| `ingest.py` | PDF/Markdown 파싱 + 텍스트 청킹 |
| `extract.py` | LLM 기반 바이오메디컬 개체/관계 추출 |
| `graph.py` | NetworkX 지식 그래프 + JSON 영구 저장 |
| `vectorstore.py` | ChromaDB 벡터 스토어 |
| `notes.py` | Obsidian 호환 마크다운 노트 생성 |
| `rag.py` | 그래프 강화 RAG 쿼리 엔진 |
| `config.py` | 설정 관리 (LLM, 임베딩, 경로) |

## Data Directory

```
data/
├── papers/           # 원본 PDF
├── graph/            # knowledge_graph.json
├── chroma_db/        # ChromaDB 벡터 저장소
└── vault/            # Obsidian vault
    ├── papers/       # 논문별 노트 ([[wikilink]] 포함)
    ├── entities/     # 개체별 노트 (유전자, pathway 등)
    └── _index.md     # 전체 색인
```

## LLM 설정

`.env`에서 provider 선택:

```env
LLM_PROVIDER=openai          # or anthropic
LLM_MODEL=gpt-4o-mini        # or claude-sonnet-4-20250514
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Conda Environment

```bash
conda activate paper_rag      # Python 3.11
```
