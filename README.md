# Qforia

Query fan-out simulator for AI Overviews / AI Mode (iPullRank).

## What's new in v2

- **Gemini 3.8 Flash** by default (`google-genai` Interactions API, schema-enforced JSON), with a thinking-level control, retries and an optional fallback model.
- **Multiple fan-out passes.** The same prompt runs *N* times (default 5). Each query gets **Freq** (how many passes produced it, exact-text match after lowercasing and stripping punctuation) and **Weight** (Freq ÷ successful passes).
- **Fan-out diagram.** Seed → cluster (or type) → queries with `Nx` badges. Edge thickness follows weight.
- **EmbeddingGemma 2 clusters** *(currently switched off — set `CLUSTERING_ENABLED = True` in `qforia.py` and uncomment the clustering packages in `requirements.txt`)*. Unique queries are embedded locally with [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) (Clustering prompt). KMeans picks k by silhouette (max 8), the map is UMAP / t-SNE / PCA, and Gemini names each cluster.
- Discovery curve per pass, CSV exports (queries, selected rows, raw passes, clusters), SQLite history you can reopen, and optional FetchSERP rank lookups.

## Run

```bash
cd streamlit
pip install -r requirements.txt        # CPU-only: install torch/torchvision from the CPU index first (see file)
streamlit run qforia.py
```

Set `GEMINI_API_KEY` (env var or `.streamlit/secrets.toml`) or paste the key in the sidebar.
With clustering on, the first run downloads EmbeddingGemma 2 (~1.5 GB) from Hugging Face. CPU works fine; float32 needs ~4 GB RAM, while bfloat16 needs ~1.6 GB.

## Storage

Runs are saved to `streamlit/qforia.sqlite`. v2 adds the `fanout_passes`, `fanout_queries` and `fanout_clusters` tables and new columns on `runs`. The v1 tables and rows are untouched and can still be exported from the History tab.
