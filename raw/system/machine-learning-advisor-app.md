# MachineLearningAdvisor — Implementation Reference

**Built:** 2026-04-16  
**Location:** `/Users/macbook/Documents/MachineLearningAdvisor`  
**Deployed:** shinyapps.io (karpeles account) as "MachineLearningAdvisor"

---

## What It Is

A Python Shiny app that wraps the llm-wiki knowledge base in a conversational interface. It uses **hybrid retrieval** — combining ChromaDB vector search with keyword index search — and synthesizes answers with gpt-5.4-mini.

Primary use case: paste a full Kaggle competition description, receive a structured ML strategy document.

---

## Architecture

```
User query
    ├── embed(query) → ChromaDB cosine search → top 12 chunks (semantic)
    └── keyword_score(query, wiki/index.md) → top 5 full pages (index)
                    ↓
    Merge + deduplicate (by source file)
                    ↓
    gpt-5.4-mini synthesis (8K output budget)
                    ↓
    Streamed response + sources sidebar
```

---

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Shiny entry point; `ui.page_navbar` with Advisor/Ingest/Browse tabs |
| `config.py` | All paths, model names, chunk settings, embed dir lists |
| `embeddings.py` | ChromaDB init/reset, chunking, `get_embeddings`, `search` |
| `wiki_search.py` | Index parser, keyword scorer, `resolve_page_path` |
| `advisor.py` | `retrieve()` + `build_messages()`; Competition Strategy Mode prompt |
| `ingestion.py` | Full ingest pipeline: save → chunk/embed → LLM wiki update → re-embed |
| `www/styles.css` | GitHub dark theme; flex layouts for all three tabs |
| `scripts/embed_all.py` | One-time full rebuild of ChromaDB from all source dirs |
| `scripts/fix_wiki_links.py` | Repaired 101 broken wikilinks before initial embed |

---

## ChromaDB

- **Collection:** `ml_wiki`
- **Path:** `db/chroma/`
- **Embedding model:** `text-embedding-3-small` (1536 dims)
- **Distance:** cosine similarity
- **Initial build:** 948 chunks, 102 documents, 51.8 seconds
- **Chunk strategy:** split at `##` section headers; sliding window (500 tok, 100 overlap) for large sections
- **Chunk metadata:** `file_path`, `category`, `title`, `tags`, `section`, `chunk_index`, `doc_type`

---

## Models

| Use | Model |
|-----|-------|
| Embeddings | `text-embedding-3-small` |
| Synthesis (default) | `gpt-5.4-mini` |
| Synthesis (optional) | `gpt-5.4` |
| Ingest agent | `gpt-5.4-mini` |

Model toggle in navbar; defaults to `gpt-5.4-mini`.

---

## Retrieval Details

**Semantic search:** ChromaDB cosine search, top 12 chunks, deduped by source file  
**Index search:** TF-IDF-like term overlap on index.md table; top 5 pages loaded in full  
**Context budget:** ~8K tokens; sources ordered by combined score  
**Context format:**
```
=== SEMANTIC SEARCH RESULTS ===
[Source: wiki/concepts/ensembling-strategies.md | Score: 0.91]
...

=== INDEX-MATCHED PAGES ===
[Source: wiki/strategies/kaggle-meta-strategy.md | Match: keyword]
...
```

---

## Competition Strategy Mode

Detects when user pastes a full competition description. System prompt activates structured response:
- Competition Analysis (task type, metric, constraints)
- Recommended Strategy (phase-by-phase)
- Data & Feature Engineering
- Training & Validation
- What NOT To Do
- Jason's Relevant Prior Work

---

## Ingestion Pipeline

1. Save raw file to `raw/<category>/`
2. Chunk + embed → ChromaDB
3. Call gpt-5.4-mini ingest agent (JSON: `{updates, creates, index_entry, log_entry}`)
4. Write/update wiki pages on disk
5. Append to `wiki/index.md` and `wiki/log.md`
6. Re-embed updated wiki pages

---

## Tabs

**Advisor** — chat with streaming responses; sources sidebar shows which chunks/pages contributed  
**Ingest** — upload .md/.txt/.pdf; category dropdown; AI updates wiki automatically  
**Browse** — searchable index table; click any row to view full document in right panel

---

## Deployment

```bash
echo "3.11" > .python-version
rsconnect write-manifest shiny . --overwrite
# Edit manifest.json: change python version to "3.11.0"
rsconnect deploy manifest manifest.json --name karpeles --title MachineLearningAdvisor --new
```

ChromaDB `db/chroma/` must be included in deployment (bundled snapshot, read-only after deploy).

---

## Known Issues / Notes

- ChromaDB collection is read-only on shinyapps.io (no persistent writes between sessions)
- Ingest tab works locally but wiki changes don't persist after session ends on deployed version
- `max_completion_tokens` (not `max_tokens`) required for gpt-5.4 API calls
- Integer tags in YAML frontmatter required `str()` coercion during chunking
- `Chat()` object must be instantiated inside `server()` function (session-scoped)
