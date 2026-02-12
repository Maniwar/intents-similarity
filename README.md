# Intent Similarity Analyzer with Semantic Search

A Streamlit application for identifying confusing intents, auditing datasets, and finding where to add new phrases in intent classification systems. All processing runs locally — no data leaves your machine.

## Features

### Similarity Analysis
- **Intent-to-intent similarity matrix** with interactive heatmap
- **Intent Health Dashboard** — composite score (0–100) combining cohesion, separation, keyword diversity, and phrase count
- **t-SNE scatter plot** — 2-D embedding projection for visualizing phrase clusters
- **Phrase-level conflict detection** — FAISS-accelerated (optional) or brute-force fallback
- **TF-IDF keyword analysis** — unigrams/bigrams with multilingual stop-word support
- **Actionable recommendations** — merge, differentiate, or rephrase suggestions ranked by priority
- **Cross-intent duplicate detection**

### Semantic Search & Audit
- **Single-phrase search** with similarity scores against all phrases in the dataset
- **Batch search** — audit dozens of phrases at once with automatic action classification:
  - `DUPLICATE` (>98%), `ADD` (85–98%), `REVIEW` (70–85%), `NEW_INTENT` (<70%)
- **Intent distribution analysis** and **search history tracking**
- **Export** to Excel, JSON, or plain text

### Project Management
- Save and load analysis sessions locally (`~/.intent_analyzer_projects/`)
- Persists embeddings, search history, and model metadata across restarts

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input Data Format

Upload a CSV where each **column** is an intent and each **row** contains phrases for that intent. Empty cells are fine.

| greeting | farewell | book_flight |
|---|---|---|
| hello | goodbye | I want to book a flight |
| hi there | see you later | book me a ticket |
| good morning | bye | reserve a flight |

## Supported Models

### Sentence Transformers

| Model | Size | Description |
|---|---|---|
| `all-MiniLM-L6-v2` | 80 MB | Ultra-fast, English only — 14k sent/sec |
| `all-mpnet-base-v2` | 420 MB | Best quality-speed balance, English only |
| `paraphrase-multilingual-mpnet-base-v2` **(default)** | 1.1 GB | 50+ languages, excellent cross-lingual quality |
| `BAAI/bge-base-en-v1.5` | 440 MB | SOTA English, instruction-aware embeddings |
| `BAAI/bge-m3` | 2.3 GB | 100+ languages, multi-granularity |
| `intfloat/e5-base-v2` | 440 MB | Balanced performer, English only |
| `intfloat/multilingual-e5-base` | 1.1 GB | 100+ languages |
| `nomic-ai/nomic-embed-text-v1.5` | 550 MB | Top BEIR scores, multimodal-ready |
| `Alibaba-NLP/gte-base-en-v1.5` | 440 MB | Angle-optimized, strong retrieval |
| `LaBSE` | 1.8 GB | 109 languages, cross-lingual search |

### Base Transformers

`xlm-roberta-large` and `xlm-roberta-base` with configurable pooling (mean or CLS).

## Configuration

**Thresholds** (adjustable in sidebar):
- Intent similarity threshold (default 0.85) — flags confusing intent pairs
- Phrase similarity threshold (default 0.90) — reports individual phrase conflicts

**Performance tuning:**
- Embedding batch size (8–512)
- Phrase similarity chunk size (64–1024)
- Mixed precision on GPU
- Force CPU mode for troubleshooting

Memory limits auto-scale based on available system RAM.

## Export

Analysis results export as an Excel workbook with sheets for:
1. Summary metrics
2. Confusing intent pairs
3. Priority actions (Critical / High / Medium)
4. All phrase conflicts
5. Intent similarity matrix

Search results export to Excel, JSON, or plain text.

## Project Structure

```
├── app.py                     # Main Streamlit app
├── requirements.txt
├── core/
│   ├── embeddings.py          # Model loading & encoding
│   ├── analysis.py            # Similarity, health scoring, t-SNE
│   ├── search.py              # Semantic search (single & batch)
│   ├── keywords.py            # TF-IDF keyword extraction
│   ├── recommendations.py     # Priority action generation
│   └── memory.py              # Device detection & memory scaling
├── ui/
│   ├── sidebar.py             # Configuration sidebar
│   ├── tab_similarity.py      # Similarity analysis tab
│   ├── tab_search.py          # Search & audit tab
│   └── components.py          # Reusable UI components
└── utils/
    ├── data_loader.py         # CSV loading & validation
    ├── export.py              # Excel/JSON/CSV export
    └── persistence.py         # Project save/load
```

## Requirements

- Python 3.10+
- PyTorch, Sentence Transformers, Transformers
- Streamlit, Plotly, Pandas, NumPy, scikit-learn
- Optional: `faiss-cpu` for ~10x faster phrase conflict detection
