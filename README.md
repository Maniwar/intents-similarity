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

## Buttons & Actions Reference

### Project Management (Sidebar)

| Button | What it does |
|---|---|
| **Save Project** | Saves the current session (dataset, embeddings, search history, model metadata) to `~/.intent_analyzer_projects/` so you can resume later |
| **Load Project** | Restores a previously saved project, including all analysis results |
| **Delete Project** | Permanently removes a saved project from disk |

### Model & Data Preparation

| Button | What it does |
|---|---|
| **Load Model** | Downloads (first time) and loads the selected sentence-transformer or base-transformer model into memory |
| **Remove Duplicates from Dataset** | Removes duplicate phrases within each intent column |
| **Remove Cross-Intent Duplicates** | Removes phrases that appear in more than one intent, keeping only the first occurrence |

### Similarity Analysis Tab

| Button | What it does |
|---|---|
| **Run Analysis** | Computes embeddings, builds the intent similarity matrix, detects phrase-level conflicts, and generates all recommendations |
| **Set as Baseline** | Snapshots current analysis results so you can compare before/after when you re-run analysis after editing the dataset |
| **Reset Baseline** | Clears the saved baseline snapshot |

### Phrase Generation (Similarity Tab)

| Button | What it does |
|---|---|
| **Generate Phrases** | Uses the loaded model to produce paraphrase variations of phrases for a selected intent |
| **Add Generated Phrases to Dataset** | Appends the generated phrases into the working dataset |
| **Clear Generated Phrases** | Discards generated phrases for the selected intent without adding them |

### Similarity Tab — Downloads

| Button | File | Contents |
|---|---|---|
| **Download Cleaned Dataset** | `cleaned_intents.csv` | Dataset after duplicate removal |
| **Download ALL Exact Duplicates** | `exact_duplicates.csv` | Every exact-match duplicate found across intents |
| **Download Full Confusion Report** | `phrase_confusion_report.csv` | All phrase-level conflicts with similarity scores |
| **Download All Priority Actions** | `priority_actions.csv` | Actionable recommendations (merge / differentiate / rephrase) ranked Critical → High → Medium |
| **Download All Phrase-Level Actions** | `phrase_level_actions.csv` | Per-phrase conflict recommendations with suggested actions |
| **Generate Complete Excel Workbook** / **Download Complete Excel Workbook** | `intent_analysis_complete.xlsx` | Multi-sheet workbook: Executive Summary, Confusing Pairs, Priority Actions, Phrase-Level Recommendations, Intent Similarity Matrix |
| **Download Modified Dataset** | `modified_intents.csv` | The current working dataset as-is, reflecting every change made during the session (duplicate removals, added generated phrases, etc.) |
| **Generate Full Report** / **Download Full Report** | `intent_analysis_report.txt` | Plain-text report with executive summary, key findings, methodology, and improvement recommendations |

### Semantic Search & Audit Tab

| Button | What it does |
|---|---|
| **Search & Analyze** | Runs a semantic search for a single query phrase against the entire dataset and displays ranked results |
| **Run Batch Audit** | Searches multiple phrases at once and classifies each as `DUPLICATE` (>98%), `ADD` (85–98%), `REVIEW` (70–85%), or `NEW_INTENT` (<70%) |
| **Save to Search History** | Stores the current search results in session history for later reference |
| **Export History** | Shows a download button for the full search history |
| **Clear History** | Deletes all saved search history |

### Search Tab — Downloads

| Button | File | Contents |
|---|---|---|
| **Download as CSV** | `semantic_search_results_<timestamp>.csv` | Search results with phrase, intent, and similarity scores |
| **Download Excel Report** | `semantic_search_audit_<timestamp>.xlsx` | Multi-sheet workbook: Search Metadata, Full Results, Intent Distribution, Top 10 Matches |
| **Download as JSON** | `semantic_search_results_<timestamp>.json` | JSON export of search results including query metadata |
| **Download History CSV** | `search_history.csv` | Complete search history with queries, thresholds, and result counts |
| **Download Batch Audit Report** | `batch_audit_<timestamp>.csv` | Summary of batch audit results with action classifications and recommendations |

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
