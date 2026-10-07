"""
Qforia — Query Fan-Out Simulator for AI Surfaces (iPullRank)

v2.0
- Gemini 3.8 Flash (default) through the google-genai Interactions API, with
  schema-enforced JSON output, retries, and an optional fallback model.
- Multiple independent fan-out passes per seed query. Every synthetic query gets a
  frequency count (how many passes produced it, after exact-text normalization)
  and a weight (frequency / successful passes).
- Fan-out diagram: seed -> cluster (or type) -> synthetic queries with Nx badges.
- EmbeddingGemma 2 (google/embeddinggemma-2) embeddings, automatic clustering and a
  2-D semantic map of the synthetic queries.
- Single or bulk input, SQLite history (re-open any past run), optional FetchSERP
  rank lookups.

Run:  streamlit run streamlit/qforia.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from google import genai
except ImportError:  # pragma: no cover - surfaced in the UI
    genai = None

APP_VERSION = "2.0.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
# Constants
# =========================================================================
GEMINI_MODELS = [
    "gemini-3.8-flash",        # latest flagship (default)
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
FALLBACK_CHOICES = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "(none)"]
THINKING_CHOICES = ["model default", "low", "medium", "high"]

EMBED_MODEL_ID = "google/embeddinggemma-2"
MAX_CLUSTERS = 8  # categorical palette has 8 validated slots

# EmbeddingGemma 2 clustering is switched off for now: it downloads a ~1.5 GB model and needs
# torch, which is slow on Streamlit Community Cloud (and its 2.7 GB RAM cap). To turn it back on,
# set this to True and uncomment the clustering packages in requirements.txt.
CLUSTERING_ENABLED = False

QUERY_TYPES = [
    "reformulation", "related", "implicit",
    "comparative", "entity_expansion", "personalized",
]
TYPE_ALIASES = {
    "reformulations": "reformulation",
    "related_queries": "related", "related_query": "related",
    "implicit_queries": "implicit", "implicit_query": "implicit",
    "comparative_queries": "comparative", "comparison": "comparative", "comparisons": "comparative",
    "entity_expansions": "entity_expansion", "entity": "entity_expansion",
    "personalised": "personalized", "personalized_queries": "personalized",
}

ALLOWED_FORMATS = [
    "web_article", "faq_page", "how_to_steps", "comparison_table", "buyers_guide", "checklist",
    "product_spec_sheet", "glossary/definition", "pricing_page", "review_roundup",
    "tutorial_video/transcript", "podcast_transcript", "code_samples/docs", "api_reference",
    "calculator/tool", "dataset", "image_gallery", "map/local_pack", "forum/qna",
    "pdf_whitepaper", "case_study", "press_release", "interactive_widget",
]

# Categorical palette (fixed order, validated for CVD on adjacent pairs).
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
# Secondary encoding for the scatter (identity is never color-only).
MARKER_SYMBOLS = ["circle", "diamond", "square", "triangle-up", "x", "star", "hexagon", "triangle-down"]
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}
NEUTRAL = {"light": "#64748b", "dark": "#94a3b8"}

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
AUTH_STATUS = {401, 403}

FETCHSERP_RANKING = "https://www.fetchserp.com/api/v1/ranking"


@dataclass
class RunConfig:
    mode: str
    model: str
    fallback_model: str | None
    thinking_level: str
    passes: int
    workers: int
    cluster_enabled: bool
    name_clusters: bool
    k_override: int
    projection: str
    embed_dim: int
    embed_precision: str
    serp_enabled: bool = False
    fetchserp_key: str = ""
    serp_domain: str = ""
    serp_engine: str = "google"
    serp_country: str = "us"
    serp_pages: int = 1
    serp_delay: float = 0.2


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# =========================================================================
# Prompting
# =========================================================================
def build_fanout_prompt(q: str, mode: str) -> str:
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"you must decide on an optimal number of queries to generate. "
            f"This number must be at least {min_queries_simple}. "
            f"For a straightforward query, generate around {min_queries_simple}-{min_queries_simple + 2}. "
            f"If the query has a few distinct aspects or common follow-ups, aim for "
            f"{min_queries_simple + 3}-{min_queries_simple + 5}. "
            f"Provide brief reasoning for why you chose this number."
        )
    else:
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"you must decide on an optimal number of queries to generate. "
            f"This number must be at least {min_queries_complex}. "
            f"For multifaceted queries that span comparisons, procedures, specs, or trade-offs, "
            f"generate {min_queries_complex + 5}-{min_queries_complex + 10} or more. "
            f"Provide brief reasoning for your number."
        )

    routing_note = (
        "For EACH expanded query, also identify the most likely CONTENT TYPE / FORMAT the routing system would "
        "prefer for retrieval and synthesis (e.g., a how-to should route to 'how_to_steps' or a video transcript; "
        "comparisons to 'comparison_table' or 'buyers_guide'). Choose exactly ONE label from this fixed list:\n"
        + ", ".join(ALLOWED_FORMATS)
        + ".\nReturn it in a field named 'routing_format' and give a short 'format_reason' (1 sentence)."
    )

    return (
        "You are simulating Google's AI Mode query fan-out for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        "Your first task is to determine the total number of queries to generate and the reasoning for this number:\n"
        f"{num_queries_instruction}\n\n"
        "Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.\n"
        "Each of the following transformation types MUST be represented at least once, if the total allows:\n"
        "1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n"
        "5. Entity Expansions\n6. Personalized Queries\n\n"
        "Write each synthetic query the way a searcher would type it (lowercase is fine, no numbering).\n"
        "The 'user_intent' field is one sentence describing what the searcher wants from that query. "
        "The 'reasoning' field explains why that query was generated (tie it to the original query, its type, "
        "and user intent). Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        f"{routing_note}\n\n"
        "Return only a JSON object that matches the response schema: generation_details "
        "(target_query_count, reasoning_for_count) and expanded_queries (query, type, user_intent, reasoning, "
        "routing_format, format_reason)."
    )


FANOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "generation_details": {
            "type": "object",
            "properties": {
                "target_query_count": {"type": "integer"},
                "reasoning_for_count": {"type": "string"},
            },
            "required": ["target_query_count", "reasoning_for_count"],
        },
        "expanded_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "type": {"type": "string", "enum": QUERY_TYPES},
                    "user_intent": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "routing_format": {"type": "string", "enum": ALLOWED_FORMATS},
                    "format_reason": {"type": "string"},
                },
                "required": ["query", "type", "user_intent", "reasoning", "routing_format", "format_reason"],
            },
        },
    },
    "required": ["generation_details", "expanded_queries"],
}

CLUSTER_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer"},
                    "label": {"type": "string"},
                },
                "required": ["cluster_id", "label"],
            },
        }
    },
    "required": ["clusters"],
}


def build_cluster_label_prompt(seed: str, groups: dict[int, list[str]]) -> str:
    lines = [f"[{cid}] " + " | ".join(qs) for cid, qs in sorted(groups.items())]
    return (
        "You are labeling clusters of synthetic search queries produced by a query fan-out for the seed query "
        f"\"{seed}\".\n"
        "For each cluster, write a concise topical label of 2-5 words that names the sub-intent the queries share "
        "and distinguishes it from the other clusters. Lowercase, no quotes, no trailing punctuation, and do not "
        "just repeat the seed query.\n\n"
        "Clusters:\n" + "\n".join(lines) + "\n\n"
        "Return JSON with one entry per cluster_id."
    )


# =========================================================================
# Gemini client (google-genai, Interactions API with generate_content fallback)
# =========================================================================
def make_client(api_key: str):
    if genai is None:
        raise RuntimeError("The google-genai package is not installed. Run: pip install -U google-genai")
    base_url = os.environ.get("GEMINI_BASE_URL")  # optional proxy / gateway
    if base_url:
        return genai.Client(api_key=api_key, http_options={"base_url": base_url})
    return genai.Client(api_key=api_key)


def status_code_of(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def parse_json_lenient(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            return json.loads(t[start:end + 1])
        raise


def _usage_dict(u, legacy: bool = False) -> dict:
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    if legacy:
        return {
            "input_tokens": getattr(u, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(u, "candidates_token_count", 0) or 0,
            "thought_tokens": getattr(u, "thoughts_token_count", 0) or 0,
        }
    return {
        "input_tokens": getattr(u, "total_input_tokens", 0) or 0,
        "output_tokens": getattr(u, "total_output_tokens", 0) or 0,
        "thought_tokens": getattr(u, "total_thought_tokens", 0) or 0,
    }


def _raw_call(client, model: str, prompt: str, schema: dict, thinking_level: str, stats: dict) -> tuple[str, dict]:
    """One Gemini request (plus at most one parameter-stripping retry on a 400)."""
    if hasattr(client, "interactions"):
        kwargs = {
            "model": model,
            "input": prompt,
            "store": False,
            "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
        }
        if thinking_level in ("low", "medium", "high"):
            kwargs["generation_config"] = {"thinking_level": thinking_level}
        for _ in range(3):
            stats["calls"] = stats.get("calls", 0) + 1
            try:
                it = client.interactions.create(**kwargs)
                text = getattr(it, "output_text", None) or ""
                status = getattr(it, "status", None)
                if not text and status not in (None, "completed"):
                    raise RuntimeError(f"Interaction ended with status '{status}' and no text output")
                return text, _usage_dict(getattr(it, "usage", None))
            except Exception as e:  # strip params some models reject, then retry once
                msg = str(e).lower()
                if status_code_of(e) == 400 and "generation_config" in kwargs and "thinking" in msg:
                    kwargs.pop("generation_config")
                    continue
                if status_code_of(e) == 400 and "store" in kwargs and "store" in msg:
                    kwargs.pop("store")
                    continue
                raise
        raise RuntimeError("Gemini request failed after parameter fallbacks")

    # Older google-genai SDKs without the Interactions API
    config = {"response_mime_type": "application/json", "response_json_schema": schema}
    if thinking_level in ("low", "medium", "high"):
        config["thinking_config"] = {"thinking_level": thinking_level}
    stats["calls"] = stats.get("calls", 0) + 1
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    return resp.text or "", _usage_dict(getattr(resp, "usage_metadata", None), legacy=True)


def call_gemini_json(client, model: str, prompt: str, schema: dict, thinking_level: str,
                     stats: dict, max_attempts: int = 3) -> tuple[dict, dict]:
    """Structured JSON call with exponential backoff on transient errors / malformed JSON."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            text, usage = _raw_call(client, model, prompt, schema, thinking_level, stats)
            data = parse_json_lenient(text)
            if not isinstance(data, dict):
                raise ValueError("Model returned JSON that is not an object")
            return data, {"model": model, "latency_s": time.perf_counter() - t0, "usage": usage, "attempts": attempt}
        except (json.JSONDecodeError, ValueError) as e:
            last_exc = e  # truncated / malformed output -> try again
        except Exception as e:
            last_exc = e
            code = status_code_of(e)
            if code is not None and code not in RETRYABLE_STATUS:
                break
        if attempt < max_attempts:
            time.sleep(min(20.0, 2 ** attempt + random.random()))
    assert last_exc is not None
    raise last_exc


# =========================================================================
# Fan-out passes and aggregation
# =========================================================================
def normalize_query(q: str) -> str:
    """Exact-match key: case, quotes, punctuation and whitespace are ignored."""
    q = unicodedata.normalize("NFKC", str(q)).lower().strip()
    q = re.sub(r"[\"'`‘’“”]", "", q)
    q = re.sub(r"[^\w\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def clean_item(obj: dict) -> dict:
    qtype = str(obj.get("type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    qtype = TYPE_ALIASES.get(qtype, qtype) or "related"
    return {
        "query": re.sub(r"\s+", " ", str(obj.get("query", ""))).strip(),
        "type": qtype,
        "user_intent": str(obj.get("user_intent", "")).strip(),
        "reasoning": str(obj.get("reasoning", "")).strip(),
        "routing_format": str(obj.get("routing_format", "")).strip(),
        "format_reason": str(obj.get("format_reason", "")).strip(),
    }


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def run_one_pass(client, seed: str, cfg: RunConfig, pass_index: int) -> dict:
    stats = {"calls": 0}
    rec = {
        "pass_index": pass_index, "model_used": None, "fallback": False,
        "target_query_count": None, "reasoning_for_count": "", "queries": [],
        "latency_s": None, "attempts": 0, "usage": {}, "error": None, "raw_json": None,
    }
    prompt = build_fanout_prompt(seed, cfg.mode)
    models = [cfg.model] + ([cfg.fallback_model] if cfg.fallback_model and cfg.fallback_model != cfg.model else [])
    errors = []
    for i, m in enumerate(models):
        try:
            data, meta = call_gemini_json(client, m, prompt, FANOUT_SCHEMA, cfg.thinking_level, stats)
            details = data.get("generation_details") or {}
            items = [clean_item(o) for o in (data.get("expanded_queries") or []) if isinstance(o, dict)]
            items = [x for x in items if x["query"]]
            if not items:
                raise ValueError("Model returned no expanded_queries")
            rec.update(
                model_used=m, fallback=i > 0,
                target_query_count=_to_int(details.get("target_query_count")),
                reasoning_for_count=str(details.get("reasoning_for_count") or ""),
                queries=items, latency_s=meta["latency_s"], usage=meta["usage"],
                raw_json=json.dumps(data, ensure_ascii=False), error=None,
            )
            break
        except Exception as e:
            errors.append(f"{m}: {type(e).__name__}: {e}")
            rec["error"] = " || ".join(errors)
            if status_code_of(e) in AUTH_STATUS:
                break  # bad key — a fallback model won't help
    rec["attempts"] = stats["calls"]
    return rec


def run_passes(client, seed: str, cfg: RunConfig, on_done=None) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(cfg.workers, cfg.passes))) as pool:
        futures = {pool.submit(run_one_pass, client, seed, cfg, i): i for i in range(1, cfg.passes + 1)}
        for fut in as_completed(futures):
            rec = fut.result()
            results.append(rec)
            if on_done:
                on_done(rec, len(results))
    return sorted(results, key=lambda r: r["pass_index"])


def aggregate_passes(passes: list[dict]) -> pd.DataFrame:
    ok = [p for p in passes if not p["error"]]
    n_ok = len(ok)
    agg: dict[str, dict] = {}
    for p in ok:
        seen_this_pass: set[str] = set()
        for rank, it in enumerate(p["queries"], start=1):
            key = normalize_query(it["query"])
            if not key:
                continue
            a = agg.get(key)
            if a is None:
                a = agg[key] = {
                    "forms": Counter(), "types": Counter(), "formats": Counter(),
                    "intent": "", "reasoning": "", "format_reason": "",
                    "passes": set(), "ranks": [], "mentions": 0, "first_pass": p["pass_index"],
                }
            a["mentions"] += 1
            a["forms"][it["query"]] += 1
            a["types"][it["type"]] += 1
            if it["routing_format"]:
                a["formats"][it["routing_format"]] += 1
            a["intent"] = a["intent"] or it["user_intent"]
            a["reasoning"] = a["reasoning"] or it["reasoning"]
            a["format_reason"] = a["format_reason"] or it["format_reason"]
            if key not in seen_this_pass:
                seen_this_pass.add(key)
                a["passes"].add(p["pass_index"])
                a["ranks"].append(rank)

    rows = []
    for key, a in agg.items():
        freq = len(a["passes"])
        rows.append({
            "query": a["forms"].most_common(1)[0][0],
            "type": a["types"].most_common(1)[0][0],
            "user_intent": a["intent"],
            "reasoning": a["reasoning"],
            "routing_format": a["formats"].most_common(1)[0][0] if a["formats"] else "",
            "format_reason": a["format_reason"],
            "frequency": freq,
            "weight": freq / n_ok if n_ok else 0.0,
            "mentions": a["mentions"],
            "avg_rank": float(np.mean(a["ranks"])) if a["ranks"] else None,
            "first_pass": a["first_pass"],
            "passes_seen": sorted(a["passes"]),
            "type_votes": dict(a["types"]),
            "format_votes": dict(a["formats"]),
            "query_norm": key,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["frequency", "avg_rank"], ascending=[False, True]).reset_index(drop=True)
    for col, default in (("cluster_id", -1), ("cluster_label", ""), ("x", np.nan), ("y", np.nan)):
        df[col] = default
    return df


def saturation_curve(passes: list[dict]) -> pd.DataFrame:
    seen: set[str] = set()
    rows = []
    for p in sorted(passes, key=lambda r: r["pass_index"]):
        if p["error"]:
            continue
        keys = {normalize_query(it["query"]) for it in p["queries"]} - {""}
        new = len(keys - seen)
        seen |= keys
        rows.append({"pass": p["pass_index"], "generated": len(p["queries"]), "new_unique": new, "cumulative_unique": len(seen)})
    return pd.DataFrame(rows)


# =========================================================================
# EmbeddingGemma 2 + clustering
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_embedder(model_id: str, precision: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # EmbeddingGemma 2 supports bfloat16 / float32 only (float16 can produce NaNs).
    if precision == "bfloat16" or (precision == "auto" and device == "cuda"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    kwargs = {"device": device, "model_kwargs": {"dtype": dtype}}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    model = SentenceTransformer(model_id, **kwargs)
    return model, device, str(dtype).replace("torch.", "")


def embed_queries(texts: list[str], precision: str, truncate_dim: int) -> tuple[np.ndarray, str]:
    model, device, dtype = load_embedder(EMBED_MODEL_ID, precision)
    emb = model.encode(texts, prompt_name="Clustering", batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    if truncate_dim and truncate_dim < emb.shape[1]:
        emb = emb[:, :truncate_dim]  # Matryoshka truncation (512 / 256 / 128) — one cached model serves all sizes
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)  # re-normalize after truncation / bf16
    return emb, f"{EMBED_MODEL_ID} · {emb.shape[1]}d · {device}/{dtype}"


def _center(emb: np.ndarray) -> np.ndarray:
    # The shared task prefix pushes every cosine toward ~0.9; centering spreads the space out.
    x = emb - emb.mean(axis=0, keepdims=True)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def cluster_embeddings(emb: np.ndarray, k_override: int = 0) -> tuple[np.ndarray, int, float | None]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(emb)
    if n < 3:
        return np.zeros(n, dtype=int), 1, None
    x = _center(emb)
    if k_override:
        k = int(max(1, min(k_override, n - 1, MAX_CLUSTERS)))
        if k == 1:
            return np.zeros(n, dtype=int), 1, None
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)
        try:
            sil = float(silhouette_score(x, labels, metric="cosine"))
        except ValueError:
            sil = None
        return labels, k, sil
    scored = []
    for k in range(2, min(MAX_CLUSTERS, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        scored.append((float(silhouette_score(x, labels, metric="cosine")), k, labels))
    if not scored:
        return np.zeros(n, dtype=int), 1, None
    # Highest silhouette wins; near-ties (within 0.005) go to the smaller, more readable k.
    top = max(s for s, _, _ in scored)
    sil, k, labels = min((t for t in scored if t[0] >= top - 0.005), key=lambda t: t[1])
    return labels, k, sil


def project_2d(emb: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    from sklearn.decomposition import PCA

    n = len(emb)
    x = _center(emb)
    if n < 4:
        coords = PCA(n_components=min(2, n)).fit_transform(x) if n > 1 else np.zeros((n, 2))
        if coords.shape[1] < 2:
            coords = np.hstack([coords, np.zeros((n, 2 - coords.shape[1]))])
        return coords, "PCA"
    if method == "UMAP":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_neighbors=max(2, min(15, n - 1)), min_dist=0.15, metric="cosine",
                random_state=42, init="spectral" if n > 10 else "random",
            )
            return reducer.fit_transform(x), "UMAP"
        except Exception:
            method = "t-SNE"
    if method == "t-SNE":
        try:
            from sklearn.manifold import TSNE

            perplexity = float(max(2.0, min(30.0, (n - 1) / 3)))
            return TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
                        random_state=42).fit_transform(x), "t-SNE"
        except Exception:
            pass
    return PCA(n_components=2).fit_transform(x), "PCA"


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _short(s: str, n: int = 44) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def apply_clustering(df: pd.DataFrame, seed: str, cfg: RunConfig, client, stats: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Embeds, clusters and projects the unique queries. Returns (df, clusters_df, info)."""
    info = {"embedding_model": None, "silhouette": None, "projection": None, "note": ""}
    emb, desc = embed_queries(df["query"].tolist(), cfg.embed_precision, cfg.embed_dim)
    info["embedding_model"] = desc
    labels, k, sil = cluster_embeddings(emb, cfg.k_override)
    info["silhouette"] = sil

    # Re-number clusters so 0 = most demand (sum of frequency), keeping colors stable within a run.
    tmp = pd.DataFrame({"c": labels, "f": df["frequency"].to_numpy()})
    order = tmp.groupby("c")["f"].agg(["sum", "size"]).sort_values(["sum", "size"], ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[c] for c in labels], dtype=int)

    coords, used = project_2d(emb, cfg.projection)
    info["projection"] = used
    df = df.copy()
    df["cluster_id"] = labels
    df["x"] = coords[:, 0].astype(float)
    df["y"] = coords[:, 1].astype(float)
    df["_emb"] = list(emb)

    # Medoid (most central query) per cluster -> default label
    reps: dict[int, str] = {}
    x = _center(emb)
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        centroid = x[idx].mean(axis=0)
        reps[cid] = df["query"].iloc[idx[int(np.argmax(x[idx] @ centroid))]]
    names = {cid: _short(q) for cid, q in reps.items()}

    if cfg.name_clusters and k > 1:
        groups = {
            int(cid): df[df["cluster_id"] == cid].sort_values("frequency", ascending=False)["query"].head(12).tolist()
            for cid in sorted(set(labels))
        }
        try:
            data, _ = call_gemini_json(client, cfg.model, build_cluster_label_prompt(seed, groups),
                                       CLUSTER_LABEL_SCHEMA, "low", stats, max_attempts=2)
            for c in data.get("clusters", []):
                cid = _to_int(c.get("cluster_id"))
                if cid in names and str(c.get("label", "")).strip():
                    names[cid] = _short(str(c["label"]).strip().rstrip("."), 40)
        except Exception as e:
            info["note"] = f"Cluster naming fell back to representative queries ({type(e).__name__})."
    df["cluster_label"] = df["cluster_id"].map(names)
    clusters_df = summarize_clusters(df, reps)
    return df, clusters_df, info


def summarize_clusters(df: pd.DataFrame, reps: dict[int, str] | None = None) -> pd.DataFrame:
    if df.empty or (df["cluster_id"] < 0).all():
        return pd.DataFrame()
    rows = []
    for cid, sub in df.groupby("cluster_id"):
        sub = sub.sort_values("frequency", ascending=False)
        rows.append({
            "cluster_id": int(cid),
            "label": sub["cluster_label"].iloc[0],
            "size": int(len(sub)),
            "total_frequency": int(sub["frequency"].sum()),
            "avg_weight": float(sub["weight"].mean()),
            "top_format": sub["routing_format"].mode().iat[0] if sub["routing_format"].notna().any() else "",
            "top_type": sub["type"].mode().iat[0],
            "representative": (reps or {}).get(int(cid), sub["query"].iloc[0]),
            "examples": " | ".join(sub["query"].head(3)),
        })
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


# =========================================================================
# FetchSERP
# =========================================================================
def normalize_domain(d: str) -> str:
    d = d.strip().lower()
    if d.startswith(("http://", "https://")):
        d = urlparse(d).netloc
    return d[4:] if d.startswith("www.") else d


def fetchserp_ranking_lookup(query_text: str, api_key: str, engine: str, country: str, domain: str, pages: int) -> dict:
    r = requests.get(
        FETCHSERP_RANKING,
        headers={"Authorization": f"Bearer {api_key}"},
        params={"query": query_text, "search_engine": engine, "country": country,
                "domain": domain, "pages_number": pages},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# =========================================================================
# SQLite (backward compatible with the v1 schema)
# =========================================================================
def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # ---- v1 tables (unchanged) ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            mode TEXT,
            model_name TEXT,
            input_mode TEXT,
            lookup_query TEXT,
            target_query_count INTEGER,
            reasoning_for_count TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            created_at TEXT NOT NULL,
            lookup_query TEXT NOT NULL,
            query TEXT NOT NULL,
            qtype TEXT,
            user_intent TEXT,
            reasoning TEXT,
            routing_format TEXT,
            format_reason TEXT,
            UNIQUE (lookup_query, query, routing_format) ON CONFLICT IGNORE,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            created_at TEXT NOT NULL,
            lookup_query TEXT NOT NULL,
            error TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS serp_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            query_id INTEGER,
            created_at TEXT NOT NULL,
            lookup_query TEXT NOT NULL,
            synthetic_query TEXT NOT NULL,
            search_engine TEXT,
            country TEXT,
            pages_number INTEGER,
            matched_domain TEXT,
            position INTEGER,
            result_url TEXT,
            result_title TEXT,
            site_name TEXT,
            raw_json TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id),
            FOREIGN KEY (query_id) REFERENCES queries(id)
        )""")
    # ---- v2 additions ----
    for col, decl in (
        ("passes", "INTEGER"), ("successful_passes", "INTEGER"), ("engine_requests", "INTEGER"),
        ("unique_queries", "INTEGER"), ("total_generated", "INTEGER"), ("cluster_count", "INTEGER"),
        ("thinking_level", "TEXT"), ("embedding_model", "TEXT"), ("status", "TEXT"),
        ("app_version", "TEXT"), ("meta_json", "TEXT"),
    ):
        _ensure_column(conn, "runs", col, decl)
    _ensure_column(conn, "serp_results", "fanout_query_id", "INTEGER")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fanout_passes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            pass_index INTEGER NOT NULL,
            model_used TEXT,
            fallback INTEGER,
            target_query_count INTEGER,
            reasoning_for_count TEXT,
            n_queries INTEGER,
            latency_s REAL,
            attempts INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            thought_tokens INTEGER,
            error TEXT,
            raw_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fanout_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            lookup_query TEXT NOT NULL,
            query TEXT NOT NULL,
            query_norm TEXT,
            qtype TEXT,
            user_intent TEXT,
            reasoning TEXT,
            routing_format TEXT,
            format_reason TEXT,
            frequency INTEGER,
            mentions INTEGER,
            weight REAL,
            avg_rank REAL,
            first_pass INTEGER,
            passes_seen TEXT,
            type_votes TEXT,
            format_votes TEXT,
            cluster_id INTEGER,
            cluster_label TEXT,
            x REAL,
            y REAL,
            position INTEGER,
            result_url TEXT,
            result_title TEXT,
            embedding BLOB,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fanout_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            label TEXT,
            size INTEGER,
            total_frequency INTEGER,
            avg_weight REAL,
            top_format TEXT,
            top_type TEXT,
            representative TEXT,
            examples TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )""")
    for stmt in (
        "CREATE INDEX IF NOT EXISTS idx_queries_lookup ON queries(lookup_query)",
        "CREATE INDEX IF NOT EXISTS idx_queries_run ON queries(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_serp_run ON serp_results(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_serp_lookup ON serp_results(lookup_query)",
        "CREATE INDEX IF NOT EXISTS idx_fq_run ON fanout_queries(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_fq_lookup ON fanout_queries(lookup_query)",
        "CREATE INDEX IF NOT EXISTS idx_fp_run ON fanout_passes(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_fc_run ON fanout_clusters(run_id)",
    ):
        cur.execute(stmt)
    conn.commit()


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    init_db(conn)
    return conn


def save_result(conn: sqlite3.Connection, res: dict, input_mode: str) -> int:
    now = utcnow()
    ok = [p for p in res["passes"] if not p["error"]]
    targets = [p["target_query_count"] for p in ok if p["target_query_count"]]
    df: pd.DataFrame = res["queries_df"]
    meta = {
        "fallback_passes": sum(1 for p in res["passes"] if p["fallback"]),
        "silhouette": res.get("silhouette"),
        "projection": res.get("projection"),
        "cluster_note": res.get("cluster_note", ""),
        "tokens": res.get("tokens", {}),
        "fallback_model": res.get("fallback_model"),
    }
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO runs (started_at, mode, model_name, input_mode, lookup_query, target_query_count,
               reasoning_for_count, passes, successful_passes, engine_requests, unique_queries, total_generated,
               cluster_count, thinking_level, embedding_model, status, app_version, meta_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            res["created_at"], res["mode"], res["model"], input_mode, res["seed"],
            int(round(float(np.mean(targets)))) if targets else None,
            ok[0]["reasoning_for_count"] if ok else None,
            res["passes_requested"], len(ok), res["engine_requests"], len(df),
            sum(len(p["queries"]) for p in ok), res["cluster_count"], res["thinking_level"],
            res.get("embedding_model"), res["status"], APP_VERSION, json.dumps(meta),
        ),
    )
    run_id = cur.lastrowid
    cur.executemany(
        """INSERT INTO fanout_passes (run_id, pass_index, model_used, fallback, target_query_count,
               reasoning_for_count, n_queries, latency_s, attempts, input_tokens, output_tokens,
               thought_tokens, error, raw_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run_id, p["pass_index"], p["model_used"], int(bool(p["fallback"])), p["target_query_count"],
                p["reasoning_for_count"], len(p["queries"]), p["latency_s"], p["attempts"],
                p["usage"].get("input_tokens"), p["usage"].get("output_tokens"), p["usage"].get("thought_tokens"),
                p["error"], p["raw_json"], now,
            )
            for p in res["passes"]
        ],
    )
    for p in res["passes"]:
        if p["error"]:
            cur.execute("INSERT INTO errors (run_id, created_at, lookup_query, error) VALUES (?, ?, ?, ?)",
                        (run_id, now, res["seed"], f"pass {p['pass_index']}: {p['error']}"))
    fq_ids = []
    for _, r in df.iterrows():
        emb = r.get("_emb") if "_emb" in df.columns else None
        cur.execute(
            """INSERT INTO fanout_queries (run_id, lookup_query, query, query_norm, qtype, user_intent, reasoning,
                   routing_format, format_reason, frequency, mentions, weight, avg_rank, first_pass, passes_seen,
                   type_votes, format_votes, cluster_id, cluster_label, x, y, position, result_url, result_title,
                   embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, res["seed"], r["query"], r["query_norm"], r["type"], r["user_intent"], r["reasoning"],
                r["routing_format"], r["format_reason"], int(r["frequency"]), int(r["mentions"]), float(r["weight"]),
                None if pd.isna(r["avg_rank"]) else float(r["avg_rank"]), int(r["first_pass"]),
                json.dumps(list(r["passes_seen"])), json.dumps(r["type_votes"]), json.dumps(r["format_votes"]),
                int(r["cluster_id"]), r["cluster_label"],
                None if pd.isna(r["x"]) else float(r["x"]), None if pd.isna(r["y"]) else float(r["y"]),
                None if pd.isna(r.get("position")) else _to_int(r.get("position")),
                r.get("result_url"), r.get("result_title"),
                np.asarray(emb, dtype=np.float32).tobytes() if isinstance(emb, (np.ndarray, list)) else None,
                now,
            ),
        )
        fq_ids.append(cur.lastrowid)
    for serp in res.get("serp_rows", []):
        idx = serp.pop("_row_index", None)
        cur.execute(
            """INSERT INTO serp_results (run_id, query_id, fanout_query_id, created_at, lookup_query, synthetic_query,
                   search_engine, country, pages_number, matched_domain, position, result_url, result_title,
                   site_name, raw_json)
               VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, fq_ids[idx] if idx is not None and idx < len(fq_ids) else None, now, res["seed"],
                serp["synthetic_query"], serp["engine"], serp["country"], serp["pages"], serp["domain"],
                serp["position"], serp["url"], serp["title"], serp["site_name"], serp["raw"][:1_000_000],
            ),
        )
    cdf: pd.DataFrame = res["clusters_df"]
    if not cdf.empty:
        cur.executemany(
            """INSERT INTO fanout_clusters (run_id, cluster_id, label, size, total_frequency, avg_weight,
                   top_format, top_type, representative, examples) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (run_id, int(c["cluster_id"]), c["label"], int(c["size"]), int(c["total_frequency"]),
                 float(c["avg_weight"]), c["top_format"], c["top_type"], c["representative"], c["examples"])
                for _, c in cdf.iterrows()
            ],
        )
    conn.commit()
    return run_id


def list_saved_runs(conn: sqlite3.Connection, limit: int = 500) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT id, started_at, lookup_query, mode, model_name, passes, successful_passes, unique_queries,
                  cluster_count, engine_requests, status
             FROM runs WHERE unique_queries IS NOT NULL ORDER BY id DESC LIMIT ?""",
        conn, params=(limit,),
    )


def load_saved_run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    run = pd.read_sql_query("SELECT * FROM runs WHERE id = ?", conn, params=(run_id,))
    if run.empty:
        return None
    r = run.iloc[0]
    meta = json.loads(r["meta_json"] or "{}")
    pdf = pd.read_sql_query("SELECT * FROM fanout_passes WHERE run_id = ? ORDER BY pass_index", conn, params=(run_id,))
    pdf = pdf.astype(object).where(pdf.notna(), None)  # SQL NULL -> None (not NaN)
    passes = []
    for _, p in pdf.iterrows():
        try:
            raw = json.loads(p["raw_json"]) if isinstance(p["raw_json"], str) and p["raw_json"] else {}
        except json.JSONDecodeError:
            raw = {}
        items = [clean_item(o) for o in raw.get("expanded_queries", []) if isinstance(o, dict)]
        passes.append({
            "pass_index": int(p["pass_index"]), "model_used": p["model_used"], "fallback": bool(p["fallback"]),
            "target_query_count": _to_int(p["target_query_count"]), "reasoning_for_count": p["reasoning_for_count"] or "",
            "queries": items, "latency_s": p["latency_s"], "attempts": _to_int(p["attempts"]) or 0,
            "usage": {"input_tokens": p["input_tokens"], "output_tokens": p["output_tokens"],
                      "thought_tokens": p["thought_tokens"]},
            "error": p["error"] or None, "raw_json": p["raw_json"],
        })
    qdf = pd.read_sql_query("SELECT * FROM fanout_queries WHERE run_id = ? ORDER BY frequency DESC, avg_rank ASC",
                            conn, params=(run_id,))
    qdf = qdf.rename(columns={"qtype": "type"})
    for col in ("passes_seen", "type_votes", "format_votes"):
        qdf[col] = qdf[col].apply(lambda s: json.loads(s) if s else ([] if col == "passes_seen" else {}))
    qdf = qdf.drop(columns=["embedding", "id", "run_id", "created_at", "lookup_query"], errors="ignore")
    qdf["cluster_label"] = qdf["cluster_label"].fillna("")
    qdf["cluster_id"] = qdf["cluster_id"].fillna(-1).astype(int)
    cdf = pd.read_sql_query("SELECT * FROM fanout_clusters WHERE run_id = ? ORDER BY cluster_id", conn, params=(run_id,))
    cdf = cdf.drop(columns=["id", "run_id"], errors="ignore")
    return {
        "seed": r["lookup_query"], "mode": r["mode"], "model": r["model_name"],
        "thinking_level": r["thinking_level"], "passes_requested": _to_int(r["passes"]) or len(passes),
        "passes": passes, "queries_df": qdf, "clusters_df": cdf,
        "engine_requests": _to_int(r["engine_requests"]) or 0, "status": r["status"] or "complete",
        "run_id": int(run_id), "embedding_model": r["embedding_model"], "cluster_count": _to_int(r["cluster_count"]) or 0,
        "silhouette": meta.get("silhouette"), "projection": meta.get("projection"),
        "cluster_note": meta.get("cluster_note", ""), "fallback_model": meta.get("fallback_model"),
        "tokens": meta.get("tokens", {}), "created_at": r["started_at"], "serp_enabled": qdf["position"].notna().any(),
    }


# =========================================================================
# Pipeline: one seed query
# =========================================================================
def process_seed(seed: str, cfg: RunConfig, client, log) -> dict:
    stats = {"calls": 0}
    status_line = log.empty()

    def on_pass(rec, done):
        mark = "✅" if not rec["error"] else "⚠️"
        extra = f" via fallback **{rec['model_used']}**" if rec["fallback"] else ""
        status_line.write(f"{mark} Pass {done}/{cfg.passes} finished — {len(rec['queries'])} queries{extra}")

    passes = run_passes(client, seed, cfg, on_pass)
    stats["calls"] += sum(p["attempts"] for p in passes)
    df = aggregate_passes(passes)
    ok = [p for p in passes if not p["error"]]
    res = {
        "seed": seed, "mode": cfg.mode, "model": cfg.model, "fallback_model": cfg.fallback_model,
        "thinking_level": cfg.thinking_level, "passes_requested": cfg.passes, "passes": passes,
        "queries_df": df, "clusters_df": pd.DataFrame(), "cluster_count": 0,
        "embedding_model": None, "silhouette": None, "projection": None, "cluster_note": "",
        "created_at": utcnow(), "serp_enabled": False, "serp_rows": [],
        "tokens": {
            k: int(sum((p["usage"] or {}).get(k) or 0 for p in passes))
            for k in ("input_tokens", "output_tokens", "thought_tokens")
        },
    }

    if not df.empty and cfg.cluster_enabled:
        if len(df) < 3:
            res["cluster_note"] = "Need at least 3 unique queries to cluster."
        else:
            try:
                log.write("🧬 Embedding with EmbeddingGemma 2 and clustering…")
                df, cdf, info = apply_clustering(df, seed, cfg, client, stats)
                res.update(queries_df=df, clusters_df=cdf, cluster_count=int(df["cluster_id"].nunique()),
                           embedding_model=info["embedding_model"], silhouette=info["silhouette"],
                           projection=info["projection"], cluster_note=info["note"])
            except ImportError as e:
                res["cluster_note"] = (f"Clustering skipped — missing package ({e.name}). "
                                       "Install sentence-transformers>=6.1, torch, torchvision and scikit-learn.")
            except Exception as e:
                res["cluster_note"] = f"Clustering skipped — {type(e).__name__}: {e}"

    if not df.empty and cfg.serp_enabled:
        res["serp_enabled"] = True
        df = res["queries_df"].copy()
        df["position"], df["result_url"], df["result_title"] = None, None, None
        domain = normalize_domain(cfg.serp_domain)
        prog = log.progress(0.0, text="FetchSERP rank lookups…")
        for i, (idx, row) in enumerate(df.iterrows(), start=1):
            try:
                data = fetchserp_ranking_lookup(row["query"], cfg.fetchserp_key, cfg.serp_engine,
                                                cfg.serp_country, domain, cfg.serp_pages)
                d = data.get("data", {}) or {}
                df.at[idx, "position"] = d.get("ranking")
                df.at[idx, "result_url"] = d.get("url")
                df.at[idx, "result_title"] = d.get("title")
                res["serp_rows"].append({
                    "_row_index": i - 1, "synthetic_query": row["query"], "engine": cfg.serp_engine,
                    "country": cfg.serp_country, "pages": cfg.serp_pages, "domain": domain,
                    "position": d.get("ranking"), "url": d.get("url"), "title": d.get("title"),
                    "site_name": d.get("site_name"), "raw": json.dumps(d, ensure_ascii=False),
                })
            except Exception as e:
                res["cluster_note"] = (res["cluster_note"] + f" FetchSERP error on '{_short(row['query'], 40)}': {e}").strip()
            prog.progress(i / len(df), text=f"FetchSERP rank lookups… {i}/{len(df)}")
            time.sleep(float(cfg.serp_delay))
        prog.empty()
        res["queries_df"] = df

    res["engine_requests"] = stats["calls"]
    res["status"] = "complete" if len(ok) == len(passes) else ("partial" if ok else "failed")
    return res


# =========================================================================
# UI helpers
# =========================================================================
def current_theme() -> str:
    """Best-effort server-side theme (used for Plotly accents; the diagram detects it client-side)."""
    try:
        if st.get_option("theme.base") == "dark":
            return "dark"
    except Exception:
        pass
    try:
        return "dark" if (st.context.theme.type or "light") == "dark" else "light"
    except Exception:
        return "light"


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .qf-kicker{font-size:.72rem;font-weight:700;letter-spacing:.12em;color:#7c3aed;text-transform:uppercase;margin:0 0 .15rem}
          .qf-title{font-size:1.65rem;font-weight:700;line-height:1.25;margin:0 0 .35rem}
          .qf-meta{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem;font-size:.9rem;opacity:.85}
          .qf-dot{opacity:.45}
          .qf-badge{font-size:.7rem;font-weight:700;letter-spacing:.06em;padding:.15rem .5rem;border-radius:.4rem;text-transform:uppercase;border:1px solid transparent}
          .qf-ok{color:#047857;background:rgba(16,185,129,.12);border-color:rgba(16,185,129,.35)}
          .qf-partial{color:#b45309;background:rgba(245,158,11,.14);border-color:rgba(245,158,11,.4)}
          .qf-failed{color:#b91c1c;background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.35)}
          .qf-model{color:#a21caf;background:rgba(217,70,239,.10);border-color:rgba(217,70,239,.35)}
          .qf-fallback{color:#9a3412;background:rgba(249,115,22,.12);border-color:rgba(249,115,22,.35)}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _esc(s) -> str:
    import html as _html
    return _html.escape(str(s), quote=True)


def render_header(res: dict) -> None:
    df = res["queries_df"]
    ok = sum(1 for p in res["passes"] if not p["error"])
    fallback_n = sum(1 for p in res["passes"] if p["fallback"])
    status_cls = {"complete": "qf-ok", "partial": "qf-partial", "failed": "qf-failed"}.get(res["status"], "qf-ok")
    parts = [
        f"<span>{len(df)} fan-outs</span>",
        f"<span>{ok}/{res['passes_requested']} passes</span>",
        f"<span>{res['engine_requests']} engine requests</span>",
        f"<span class='qf-badge {status_cls}'>{_esc(res['status'])}</span>",
        f"<span class='qf-badge qf-model'>{_esc(res['model'])}</span>",
    ]
    if fallback_n:
        parts.append(f"<span class='qf-badge qf-fallback'>fallback × {fallback_n}</span>")
    if res.get("cluster_count"):
        parts.insert(1, f"<span>{res['cluster_count']} clusters</span>")
    meta = " <span class='qf-dot'>•</span> ".join(parts)
    run_ref = f" · run #{res['run_id']}" if res.get("run_id") else ""
    st.markdown(
        f"<div class='qf-kicker'>Seed query{_esc(run_ref)}</div>"
        f"<div class='qf-title'>{_esc(res['seed'])}</div>"
        f"<div class='qf-meta'>{meta}</div>",
        unsafe_allow_html=True,
    )


TYPE_DISPLAY = {t: t.replace("_", " ").upper() for t in QUERY_TYPES}


def cluster_color(cid: int, theme: str) -> str:
    if cid is None or cid < 0:
        return NEUTRAL[theme]
    return PALETTE[theme][int(cid) % len(PALETTE[theme])]


def build_diagram_html(res: dict, group_by: str, min_freq: int, theme: str, key: str = "",
                       max_leaves: int = 120) -> tuple[str, int]:
    df = res["queries_df"]
    df = df[df["frequency"] >= min_freq].copy()
    has_clusters = (df["cluster_id"] >= 0).any() if not df.empty else False
    if group_by == "Cluster" and not has_clusters:
        group_by = "Type"
    n_ok = sum(1 for p in res["passes"] if not p["error"]) or 1
    hidden = max(0, len(df) - max_leaves)
    df = df.sort_values(["frequency", "avg_rank"], ascending=[False, True]).head(max_leaves)

    def leaf(r) -> dict:
        tip = (f"Intent: {r['user_intent']}\nRouting format: {r['routing_format']}\n"
               f"Seen in passes: {', '.join(str(i) for i in r['passes_seen'])} ({r['frequency']}/{n_ok})")
        if r.get("cluster_label"):
            tip += f"\nCluster: {r['cluster_label']}"
        return {
            "q": r["query"], "type": TYPE_DISPLAY.get(r["type"], str(r["type"]).upper()),
            "freq": int(r["frequency"]), "weight": float(r["weight"]),
            "ci": int(r["cluster_id"]), "tip": tip,
        }

    groups = []
    if group_by == "Cluster":
        order = (df.groupby("cluster_id")["frequency"].sum().sort_values(ascending=False).index.tolist())
        for cid in order:
            sub = df[df["cluster_id"] == cid].sort_values(["frequency", "avg_rank"], ascending=[False, True])
            groups.append({
                "label": sub["cluster_label"].iloc[0] or f"Cluster {cid + 1}",
                "meta": f"{_plural(len(sub), 'query', 'queries')} · {_plural(int(sub['frequency'].sum()), 'hit', 'hits')}",
                "ci": int(cid),
                "leaves": [leaf(r) for _, r in sub.iterrows()],
            })
    elif group_by == "Type":
        for t in QUERY_TYPES + sorted(set(df["type"]) - set(QUERY_TYPES)):
            sub = df[df["type"] == t]
            if sub.empty:
                continue
            groups.append({
                "label": TYPE_DISPLAY.get(t, str(t).upper()).title(),
                "meta": f"{_plural(len(sub), 'query', 'queries')} · {_plural(int(sub['frequency'].sum()), 'hit', 'hits')}",
                "ci": -1,
                "leaves": [leaf(r) for _, r in sub.iterrows()],
            })
    else:
        groups.append({"label": None, "meta": "", "ci": -1,
                       "leaves": [leaf(r) for _, r in df.iterrows()]})

    payload = {
        "seed": res["seed"],
        "seedMeta": f"{len(res['queries_df'])} unique · {n_ok} pass{'es' if n_ok != 1 else ''}",
        "grouped": group_by != "None",
        "groups": groups,
        "hidden": hidden,
        "theme": theme,
        "palette": PALETTE,
        "neutral": NEUTRAL,
    }

    # Height estimate (the component iframe can't auto-size); it scrolls if wrapping runs long.
    chars_per_line = 50 if payload["grouped"] else 75
    def leaf_h(l):
        return 20 * max(1, math.ceil((len(l["q"]) + 6) / chars_per_line)) + 16 + 12
    total = 0
    for g in groups:
        total += max(64 if payload["grouped"] else 0, sum(leaf_h(l) for l in g["leaves"])) + 18
    height = int(min(max(total + 90, 260), 1600))

    data_json = (json.dumps(payload, ensure_ascii=False)
                 .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    uid = "qf-" + hashlib.md5((key + data_json).encode("utf-8")).hexdigest()[:12]
    fragment = DIAGRAM_TEMPLATE.replace("__PAYLOAD__", data_json).replace("__UID__", uid)
    return fragment, height


def show_diagram(fragment: str, height: int) -> None:
    """Inline (auto-height) on Streamlit >= 1.52; sandboxed iframe on older versions."""
    import inspect

    if "unsafe_allow_javascript" in inspect.signature(st.html).parameters:
        st.html(fragment, unsafe_allow_javascript=True)
    else:
        components.html(
            "<!doctype html><html><head><meta charset='utf-8'><style>html,body{margin:0;background:transparent}"
            "</style></head><body>" + fragment + "</body></html>",
            height=height, scrolling=True,
        )


DIAGRAM_TEMPLATE = r"""
<div class="qf-diagram" id="__UID__">
<style>
  .qf-diagram{--surface:#f4f5f8;--card:#ffffff;--ink:#0f172a;--ink2:#475569;--muted:#94a3b8;--border:#e2e8f0;
        --seed:#0f172a;--seed-ink:#ffffff;--kicker:#a78bfa;
        --b1:#cde2fb;--b1i:#104281;--b2:#86b6ef;--b2i:#0d366b;--b3:#2a78d6;--b3i:#ffffff;
        font-family:Inter,"Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);}
  .qf-diagram.dark{--surface:#161a23;--card:#1f2430;--ink:#f1f5f9;--ink2:#cbd5e1;--muted:#8b97ab;--border:#2d3443;
        --seed:#0b0e14;--seed-ink:#ffffff;--kicker:#c4b5fd;
        --b1:#184f95;--b1i:#cde2fb;--b2:#256abf;--b2i:#ffffff;--b3:#3987e5;--b3i:#ffffff;}
  .qf-diagram .panel{background:var(--surface);border-radius:14px;padding:26px 22px;box-sizing:border-box;}
  .qf-diagram .tree{position:relative;display:grid;grid-template-columns:minmax(170px,240px) 1fr;column-gap:60px;align-items:center;}
  .qf-diagram svg.edges{position:absolute;left:0;top:0;overflow:visible;pointer-events:none;z-index:0;}
  .qf-diagram .seed{position:relative;z-index:1;background:var(--seed);color:var(--seed-ink);border-radius:14px;padding:18px 20px;
        text-align:center;box-shadow:0 8px 22px rgba(15,23,42,.22);}
  .qf-diagram .seed .k{font-size:11px;letter-spacing:.14em;font-weight:700;color:var(--kicker);margin-bottom:8px;}
  .qf-diagram .seed .t{font-size:15px;font-weight:650;line-height:1.4;}
  .qf-diagram .seed .m{font-size:11px;opacity:.7;margin-top:8px;}
  .qf-diagram .right{display:flex;flex-direction:column;gap:18px;position:relative;z-index:1;}
  .qf-diagram .group{display:grid;grid-template-columns:minmax(120px,180px) 1fr;column-gap:52px;align-items:center;}
  .qf-diagram .group.flat{display:block;}
  .qf-diagram .gnode{background:var(--card);border:1px solid var(--border);border-left:4px solid var(--c);border-radius:10px;
         padding:9px 12px;box-shadow:0 1px 2px rgba(15,23,42,.06);}
  .qf-diagram .gnode .gl{font-size:13px;font-weight:650;color:var(--ink);line-height:1.3;}
  .qf-diagram .gnode .gm{font-size:11px;color:var(--ink2);margin-top:3px;}
  .qf-diagram .leaves{display:flex;flex-direction:column;gap:12px;}
  .qf-diagram .leaf{display:flex;align-items:flex-start;gap:10px;cursor:default;}
  .qf-diagram .dot{width:9px;height:9px;border-radius:50%;background:var(--c);margin-top:5px;flex:none;box-shadow:0 0 0 2px var(--surface);}
  .qf-diagram .lt{font-size:14px;font-weight:600;color:var(--ink);line-height:1.4;}
  .qf-diagram .nw{white-space:nowrap;}
  .qf-diagram .badge{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;margin-left:8px;vertical-align:1px;white-space:nowrap;}
  .qf-diagram .ty{font-size:10px;letter-spacing:.12em;font-weight:650;color:var(--muted);margin-top:2px;}
  .qf-diagram .more{font-size:12px;color:var(--ink2);margin-top:14px;text-align:right;}
  .qf-diagram .leaf:hover .lt{text-decoration:underline;text-decoration-color:var(--c);text-underline-offset:3px;}
</style>
<div class="panel"><div class="tree"></div><div class="more"></div></div>
<script type="application/json" class="qf-payload">__PAYLOAD__</script>
<script>
(function(){
  const root = document.getElementById('__UID__');
  if (!root || root.dataset.ready) return;
  root.dataset.ready = '1';
  const data = JSON.parse(root.querySelector('.qf-payload').textContent);

  // Follow the page's actual theme (Streamlit can report the wrong one on first load).
  function pageIsDark(){
    try {
      const doc = (window.parent && window.parent !== window) ? window.parent.document : document;
      const app = doc.querySelector('.stApp') || doc.body;
      const m = (getComputedStyle(app).backgroundColor || '').match(/[0-9.]+/g);
      if (!m || m.length === 3 + 1 && Number(m[3]) === 0) return data.theme === 'dark';
      const lum = 0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2];
      return lum <= 128;
    } catch (e) { return data.theme === 'dark'; }
  }
  let dark = pageIsDark();
  const colorFor = ci => {
    const mode = dark ? 'dark' : 'light';
    return ci >= 0 ? data.palette[mode][ci % data.palette[mode].length] : data.neutral[mode];
  };
  const tinted = []; // [element, clusterIndex]
  root.classList.toggle('dark', dark);
  const tree = root.querySelector('.tree');
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  const seed = el('div', 'seed');
  seed.appendChild(el('div', 'k', 'SEED QUERY'));
  seed.appendChild(el('div', 't', data.seed));
  seed.appendChild(el('div', 'm', data.seedMeta));
  tree.appendChild(seed);

  const right = el('div', 'right');
  tree.appendChild(right);
  const links = []; // [fromEl, toEl, color, weight]

  function badgeStyle(w){
    if (w >= 0.67) return ['var(--b3)', 'var(--b3i)'];
    if (w >= 0.34) return ['var(--b2)', 'var(--b2i)'];
    return ['var(--b1)', 'var(--b1i)'];
  }

  data.groups.forEach(g => {
    const row = el('div', data.grouped ? 'group' : 'group flat');
    let gnode = null;
    if (data.grouped) {
      gnode = el('div', 'gnode');
      gnode.style.setProperty('--c', colorFor(g.ci)); tinted.push([gnode, g.ci]);
      gnode.appendChild(el('div', 'gl', g.label));
      gnode.appendChild(el('div', 'gm', g.meta));
      row.appendChild(gnode);
      const wavg = g.leaves.reduce((s, l) => s + l.weight, 0) / Math.max(1, g.leaves.length);
      links.push([seed, gnode, g.ci, Math.min(1, 0.35 + wavg)]);
    }
    const leaves = el('div', 'leaves');
    g.leaves.forEach(l => {
      const leaf = el('div', 'leaf');
      leaf.style.setProperty('--c', colorFor(l.ci)); tinted.push([leaf, l.ci]);
      leaf.title = l.tip;
      const dot = el('span', 'dot');
      const body = el('div');
      // keep the last word and the Nx badge together when the line wraps
      const words = l.q.split(' ');
      const last = words.pop();
      const lt = el('div', 'lt', words.length ? words.join(' ') + ' ' : '');
      const tail = el('span', 'nw', last);
      const b = el('span', 'badge', l.freq + 'x');
      const [bg, ink] = badgeStyle(l.weight);
      b.style.background = bg; b.style.color = ink;
      tail.appendChild(b);
      lt.appendChild(tail);
      body.appendChild(lt);
      body.appendChild(el('div', 'ty', l.type));
      leaf.appendChild(dot); leaf.appendChild(body);
      leaves.appendChild(leaf);
      links.push([gnode || seed, dot, data.grouped ? g.ci : l.ci, l.weight]);
    });
    row.appendChild(leaves);
    right.appendChild(row);
  });
  if (data.hidden > 0) root.querySelector('.more').textContent = '+ ' + data.hidden + ' more queries below the display limit (see table)';

  // The edge layer is created here because st.html's sanitizer strips inline svg markup.
  // (Keep this script free of a less-than sign followed by a letter or slash: the sanitizer drops such scripts.)
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'edges');
  tree.insertBefore(svg, tree.firstChild);
  function draw(){
    if (!root.isConnected) return;
    const t = tree.getBoundingClientRect();
    svg.setAttribute('width', t.width); svg.setAttribute('height', t.height);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    links.forEach(([a, b, ci, w]) => {
      const color = colorFor(ci);
      const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
      const x1 = ra.right - t.left, y1 = ra.top + ra.height / 2 - t.top;
      const x2 = rb.left - t.left - (b.classList.contains('dot') ? 2 : 0), y2 = rb.top + rb.height / 2 - t.top;
      const dx = Math.max(24, (x2 - x1) * 0.55);
      const p = document.createElementNS(NS, 'path');
      p.setAttribute('d', `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`);
      p.setAttribute('fill', 'none');
      p.setAttribute('stroke', color);
      p.setAttribute('stroke-width', (1 + 2.2 * w).toFixed(2));
      p.setAttribute('stroke-opacity', (0.28 + 0.5 * w).toFixed(2));
      p.setAttribute('stroke-linecap', 'round');
      svg.appendChild(p);
    });
  }
  requestAnimationFrame(draw);
  setTimeout(draw, 250);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);
  window.addEventListener('resize', draw);
  setInterval(() => {  // re-tint if the viewer switches theme in Streamlit's settings
    const now = pageIsDark();
    if (now === dark || !root.isConnected) return;
    dark = now; root.classList.toggle('dark', dark);
    tinted.forEach(([e, ci]) => e.style.setProperty('--c', colorFor(ci)));
    draw();
  }, 1500);
  if (window.ResizeObserver) new ResizeObserver(draw).observe(tree);
})();
</script>
</div>
"""


def cluster_map_figure(res: dict, theme: str) -> go.Figure:
    df = res["queries_df"]
    df = df[df["cluster_id"] >= 0]
    n_ok = sum(1 for p in res["passes"] if not p["error"]) or 1
    surface = SURFACE[theme]
    ink2 = "#c3c2b7" if theme == "dark" else "#52514e"
    fig = go.Figure()
    for cid in sorted(df["cluster_id"].unique()):
        sub = df[df["cluster_id"] == cid]
        label = sub["cluster_label"].iloc[0] or f"Cluster {cid + 1}"
        color = cluster_color(int(cid), theme)
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers", name=f"{label} ({len(sub)})",
            marker=dict(
                size=(10 + 16 * sub["weight"]).round(1), color=color,
                symbol=MARKER_SYMBOLS[int(cid) % len(MARKER_SYMBOLS)],
                line=dict(width=2, color=surface), opacity=0.95,
            ),
            customdata=np.stack([
                sub["query"], [label] * len(sub), sub["type"].map(lambda t: TYPE_DISPLAY.get(t, t)),
                sub["routing_format"], sub["frequency"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Cluster: %{customdata[1]}<br>Type: %{customdata[2]}"
                f"<br>Routing: %{{customdata[3]}}<br>Freq: %{{customdata[4]}}x of {n_ok} passes<extra></extra>"
            ),
        ))
        # Direct label above the cluster
        fig.add_annotation(
            x=float(sub["x"].mean()), y=float(sub["y"].max()), text=f"<b>{_esc(label)}</b>",
            showarrow=False, yshift=18, font=dict(size=12, color=ink2),
            bgcolor="rgba(14,17,23,.6)" if theme == "dark" else "rgba(255,255,255,.75)",
        )
    fig.update_layout(
        height=560, margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0, title_text=""),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hoverlabel=dict(align="left"),
    )
    return fig


def saturation_figure(sat: pd.DataFrame, theme: str) -> go.Figure:
    color = PALETTE[theme][0]
    fig = go.Figure(go.Scatter(
        x=sat["pass"], y=sat["cumulative_unique"], mode="lines+markers",
        line=dict(width=2, color=color), marker=dict(size=9, color=color, line=dict(width=2, color=SURFACE[theme])),
        customdata=np.stack([sat["new_unique"], sat["generated"]], axis=-1),
        hovertemplate="Pass %{x}<br>Unique so far: %{y}<br>New this pass: %{customdata[0]}"
                      "<br>Generated this pass: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(
        height=280, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
        xaxis=dict(title="Pass", dtick=1), yaxis=dict(title="Cumulative unique queries", rangemode="tozero"),
    )
    return fig


def google_search_url(query: str) -> str:
    return "https://www.google.com/search?q=" + quote_plus(str(query))


def results_table(res: dict) -> pd.DataFrame:
    df = res["queries_df"].copy()
    df["google"] = df["query"].map(google_search_url)
    df["weight_pct"] = (df["weight"] * 100).round(0)
    df["type_display"] = df["type"].map(lambda t: TYPE_DISPLAY.get(t, str(t).upper()))
    cols = ["query", "google", "type_display", "user_intent", "routing_format", "cluster_label",
            "frequency", "weight_pct", "passes_seen", "avg_rank"]
    if not (df["cluster_id"] >= 0).any():
        cols.remove("cluster_label")
    if res.get("serp_enabled") and "position" in df.columns:
        cols += ["position", "result_url"]
    return df[[c for c in cols if c in df.columns]]


def export_frame(res: dict) -> pd.DataFrame:
    df = res["queries_df"].drop(columns=["_emb"], errors="ignore").copy()
    df.insert(0, "lookup_query", res["seed"])
    df.insert(2, "google_search_url", df["query"].map(google_search_url))
    df["passes_seen"] = df["passes_seen"].apply(lambda v: ";".join(str(i) for i in v))
    df["type_votes"] = df["type_votes"].apply(json.dumps)
    df["format_votes"] = df["format_votes"].apply(json.dumps)
    df["passes_ok"] = sum(1 for p in res["passes"] if not p["error"])
    df["model"] = res["model"]
    return df


def passes_frame(res: dict) -> pd.DataFrame:
    rows = []
    for p in res["passes"]:
        rows.append({
            "pass": p["pass_index"], "model_used": p["model_used"] or "—", "fallback": p["fallback"],
            "target_count": p["target_query_count"], "generated": len(p["queries"]),
            "latency_s": None if p["latency_s"] is None else round(float(p["latency_s"]), 1),
            "requests": p["attempts"],
            "output_tokens": (p["usage"] or {}).get("output_tokens"),
            "thought_tokens": (p["usage"] or {}).get("thought_tokens"),
            "reasoning_for_count": p["reasoning_for_count"], "error": p["error"] or "",
        })
    return pd.DataFrame(rows)


def raw_rows_frame(res: dict) -> pd.DataFrame:
    rows = []
    for p in res["passes"]:
        for rank, it in enumerate(p["queries"], start=1):
            rows.append({"lookup_query": res["seed"], "pass": p["pass_index"], "rank": rank,
                         "model_used": p["model_used"], **it})
    return pd.DataFrame(rows)


def render_result(res: dict, key: str) -> None:
    theme = current_theme()
    df = res["queries_df"]
    render_header(res)
    if df.empty:
        errs = [p["error"] for p in res["passes"] if p["error"]]
        st.error("No synthetic queries were generated." + (f"\n\nFirst error: `{errs[0]}`" if errs else ""))
        return

    n_ok = sum(1 for p in res["passes"] if not p["error"])
    consensus = int((df["weight"] >= 0.5).sum())
    total_generated = sum(len(p["queries"]) for p in res["passes"] if not p["error"])
    targets = [p["target_query_count"] for p in res["passes"] if p["target_query_count"]]
    st.write("")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unique fan-out queries", len(df), help="After exact-text normalization across all passes.")
    c2.metric("Generated (all passes)", total_generated)
    c3.metric("Consensus queries", consensus, help="Produced in at least half of the successful passes.")
    c4.metric("Avg target count", f"{np.mean(targets):.1f}" if targets else "—",
              help="The query count the model chose for itself, averaged over passes.")
    c5.metric("Stability", f"{(df['frequency'].sum() / (len(df) * n_ok)) * 100:.0f}%" if n_ok else "—",
              help="Average weight: 100% means every pass produced the same set.")

    # ---- Fan-out diagram ----
    with st.expander("Fan-out diagram", expanded=True):
        has_clusters = bool((df["cluster_id"] >= 0).any())
        gc1, gc2 = st.columns([2, 3])
        with gc1:
            group_opts = (["Cluster"] if has_clusters else []) + ["Type", "None"]
            group_by = st.segmented_control("Group by", group_opts, default=group_opts[0], key=f"{key}_grp") or group_opts[0]
        with gc2:
            max_f = int(df["frequency"].max())
            min_freq = st.slider("Minimum frequency", 1, max(2, max_f), 1, key=f"{key}_minf",
                                 help="Hide queries that showed up in fewer passes.", disabled=max_f < 2)
        fragment, height = build_diagram_html(res, group_by, min_freq, theme, key=key)
        show_diagram(fragment, height)
        st.caption("Edge thickness and badge shade = weight (share of passes that produced the query). "
                   "Hover a query for its intent, routing format and the passes it appeared in.")

    # ---- Table ----
    st.subheader("Fan-out queries")
    tdf = results_table(res)
    colcfg = {
        "query": st.column_config.TextColumn("Query", width="large"),
        "google": st.column_config.LinkColumn("Google", display_text="Search ↗",
                                              help="Open this query in Google Search (new tab)"),
        "type_display": st.column_config.TextColumn("Type"),
        "user_intent": st.column_config.TextColumn("Intent", width="large"),
        "routing_format": st.column_config.TextColumn("Routing format"),
        "cluster_label": st.column_config.TextColumn("Cluster"),
        "frequency": st.column_config.NumberColumn("Freq", format="%dx",
                                                   help=f"Number of passes (of {n_ok}) that produced this query"),
        "weight_pct": st.column_config.ProgressColumn("Weight", min_value=0, max_value=100, format="%d%%"),
        "passes_seen": st.column_config.ListColumn("Passes"),
        "avg_rank": st.column_config.NumberColumn("Avg rank", format="%.1f",
                                                  help="Average position of the query inside each pass's list"),
        "position": st.column_config.NumberColumn("SERP pos."),
        "result_url": st.column_config.LinkColumn("Ranking URL"),
    }
    event = st.dataframe(tdf, hide_index=True, column_config=colcfg, on_select="rerun",
                         selection_mode="multi-row", key=f"{key}_tbl",
                         height=min(36 * (len(tdf) + 1) + 4, 640))
    selected = list(getattr(getattr(event, "selection", None), "rows", []) or [])

    exp = export_frame(res)
    slug = re.sub(r"[^a-z0-9]+", "-", res["seed"].lower()).strip("-")[:48] or "query"
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Download queries (CSV)", exp.to_csv(index=False).encode("utf-8"),
                       file_name=f"qforia_{slug}.csv", mime="text/csv", key=f"{key}_dl1")
    d2.download_button(f"Download selected ({len(selected)})", exp.iloc[selected].to_csv(index=False).encode("utf-8"),
                       file_name=f"qforia_{slug}_selected.csv", mime="text/csv", key=f"{key}_dl2",
                       disabled=not selected)
    d3.download_button("Download raw passes (CSV)", raw_rows_frame(res).to_csv(index=False).encode("utf-8"),
                       file_name=f"qforia_{slug}_passes.csv", mime="text/csv", key=f"{key}_dl3")
    if not res["clusters_df"].empty:
        d4.download_button("Download clusters (CSV)", res["clusters_df"].to_csv(index=False).encode("utf-8"),
                           file_name=f"qforia_{slug}_clusters.csv", mime="text/csv", key=f"{key}_dl4")

    # ---- Cluster map ----
    has_clusters = bool((df["cluster_id"] >= 0).any())
    if CLUSTERING_ENABLED or has_clusters:
        st.subheader("Semantic clusters")
    if has_clusters and df["x"].notna().any():
        sil = res.get("silhouette")
        st.caption(
            f"{res.get('embedding_model') or EMBED_MODEL_ID} · Clustering prompt · KMeans "
            f"(k={res.get('cluster_count')}{f', silhouette {sil:.2f}' if sil is not None else ''}) · "
            f"{res.get('projection') or ''} projection. Marker size = weight; marker shape = cluster."
        )
        st.plotly_chart(cluster_map_figure(res, theme), key=f"{key}_map", theme="streamlit")
        cdf = res["clusters_df"].copy()
        if not cdf.empty:
            cdf["avg_weight"] = (cdf["avg_weight"] * 100).round(0)
            st.dataframe(
                cdf.drop(columns=["cluster_id"], errors="ignore"), hide_index=True,
                column_config={
                    "label": st.column_config.TextColumn("Cluster"),
                    "size": st.column_config.NumberColumn("Queries"),
                    "total_frequency": st.column_config.NumberColumn("Total hits"),
                    "avg_weight": st.column_config.ProgressColumn("Avg weight", min_value=0, max_value=100, format="%d%%"),
                    "top_format": st.column_config.TextColumn("Top routing format"),
                    "top_type": st.column_config.TextColumn("Top type"),
                    "representative": st.column_config.TextColumn("Most central query", width="medium"),
                    "examples": st.column_config.TextColumn("Top queries", width="large"),
                },
                key=f"{key}_ctbl",
            )
    if res.get("cluster_note"):
        st.info(res["cluster_note"])
    elif CLUSTERING_ENABLED and not has_clusters:
        st.info("Clustering was turned off for this run.")

    # ---- Passes ----
    with st.expander(f"Passes ({n_ok}/{res['passes_requested']} succeeded)", expanded=False):
        sat = saturation_curve(res["passes"])
        if len(sat) > 1:
            st.caption("Discovery curve — does another pass still surface new queries?")
            st.plotly_chart(saturation_figure(sat, theme), key=f"{key}_sat", theme="streamlit")
        st.dataframe(passes_frame(res), hide_index=True, key=f"{key}_ptbl")
        tok = res.get("tokens") or {}
        if tok:
            st.caption(f"Tokens — input {tok.get('input_tokens', 0):,} · output {tok.get('output_tokens', 0):,} · "
                       f"thinking {tok.get('thought_tokens', 0):,}")


# =========================================================================
# App
# =========================================================================
st.set_page_config(page_title="Qforia", page_icon="🔍", layout="wide")
inject_css()
st.title("🔍 Qforia: Query Fan-Out Simulator for AI Surfaces")

def _default_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


with st.sidebar:
    st.header("Configuration")
    gemini_key = st.text_input("Gemini API Key", value=_default_key(), type="password")
    model_choice = st.selectbox("Gemini model", GEMINI_MODELS + ["Custom…"], index=0,
                                help="gemini-3.8-flash is the current flagship.")
    model_name = (st.text_input("Custom model id", value="gemini-3.8-flash").strip()
                  if model_choice == "Custom…" else model_choice)
    thinking_level = st.selectbox("Thinking level", THINKING_CHOICES, index=0,
                                  help="Gemini 3.x: low / medium / high. 'model default' sends nothing.")
    fallback_choice = st.selectbox("Fallback model", FALLBACK_CHOICES, index=0,
                                   help="Used for a pass only if the primary model keeps failing.")
    fallback_model = None if fallback_choice == "(none)" else fallback_choice

    st.markdown("---")
    input_mode = st.radio("Input Mode", ["Single query", "Bulk list"], horizontal=True)
    if input_mode == "Single query":
        user_query = st.text_area("Enter your query", "What's the best electric SUV for driving up mt rainier?", height=110)
        bulk_text = ""
    else:
        user_query = ""
        bulk_text = st.text_area(
            "Paste queries (one per line)",
            "best electric suv for snow\nsleep training methods for toddlers\nhow to freeze sourdough starter",
            height=160,
        )
    mode = st.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])
    passes = st.slider("Fan-out passes per query", 1, 10, 5,
                       help="Each pass is an independent Gemini call with the same prompt. "
                            "Freq = how many passes produced a query.")
    workers = st.slider("Parallel requests", 1, 10, 5)

    st.markdown("---")
    # Defaults used while clustering is switched off (see CLUSTERING_ENABLED).
    cluster_enabled, name_clusters, k_override = False, False, 0
    projection, embed_dim, embed_precision = "UMAP", 768, "auto"
    if CLUSTERING_ENABLED:
        with st.expander("Clustering (EmbeddingGemma 2)", expanded=False):
            cluster_enabled = st.checkbox("Embed & cluster synthetic queries", value=True,
                                          help="Runs google/embeddinggemma-2 locally (first run downloads ~1.5 GB).")
            name_clusters = st.checkbox("Name clusters with Gemini", value=True, disabled=not cluster_enabled,
                                        help="One extra request per seed query.")
            k_override = st.slider("Number of clusters (0 = auto)", 0, MAX_CLUSTERS, 0, disabled=not cluster_enabled)
            projection = st.selectbox("Map projection", ["UMAP", "t-SNE", "PCA"], disabled=not cluster_enabled)
            embed_dim = st.selectbox("Embedding dimensions", [768, 512, 256, 128], index=0, disabled=not cluster_enabled,
                                     help="Matryoshka truncation; 768 is full quality.")
            embed_precision = st.selectbox("Precision", ["auto", "float32", "bfloat16"], index=0,
                                           disabled=not cluster_enabled,
                                           help="auto = bfloat16 on CUDA, float32 elsewhere. float16 is not supported.")

    with st.expander("Storage", expanded=False):
        db_path = st.text_input("SQLite DB path", value=os.path.join(APP_DIR, "qforia.sqlite"))
        save_to_db = st.checkbox("Save runs to SQLite", value=True)

    with st.expander("Rank lookups (FetchSERP)", expanded=False):
        enable_serp = st.checkbox("Enable FetchSERP rank lookups", value=False)
        fetchserp_key = st.text_input("FetchSERP API Key", type="password", disabled=not enable_serp)
        target_domain = st.text_input("Domain to match (e.g., example.com)", disabled=not enable_serp)
        serp_engine = st.selectbox("Search engine", ["google", "bing", "yahoo", "duckduckgo"], disabled=not enable_serp)
        serp_country = st.text_input("Country (2-letter)", value="us", disabled=not enable_serp)
        pages_number = st.number_input("Pages to scan (1-30)", 1, 30, 1, disabled=not enable_serp)
        serp_delay = st.number_input("Delay between calls (s)", 0.0, 10.0, 0.2, step=0.1, disabled=not enable_serp)

    run_clicked = st.button("Run Fan-Out 🚀", type="primary", width="stretch")

if "results" not in st.session_state:
    st.session_state.results = []

# ---------------------------------------------------------------- run
if run_clicked:
    lookups = ([user_query.strip()] if user_query.strip() else []) if input_mode == "Single query" else \
        list(dict.fromkeys(q.strip() for q in bulk_text.splitlines() if q.strip()))
    problems = []
    if not gemini_key:
        problems.append("Enter your Gemini API key in the sidebar.")
    if not lookups:
        problems.append("Provide at least one query.")
    if enable_serp and (not fetchserp_key or not target_domain):
        problems.append("FetchSERP lookups need an API key and a domain.")
    if problems:
        for p in problems:
            st.warning(p)
    else:
        cfg = RunConfig(
            mode=mode, model=model_name, fallback_model=fallback_model, thinking_level=thinking_level,
            passes=int(passes), workers=int(workers), cluster_enabled=cluster_enabled, name_clusters=name_clusters,
            k_override=int(k_override), projection=projection, embed_dim=int(embed_dim),
            embed_precision=embed_precision, serp_enabled=enable_serp, fetchserp_key=fetchserp_key or "",
            serp_domain=target_domain or "", serp_engine=serp_engine, serp_country=serp_country,
            serp_pages=int(pages_number), serp_delay=float(serp_delay),
        )
        try:
            client = make_client(gemini_key)
        except Exception as e:
            st.error(str(e))
            st.stop()
        st.session_state.results = []
        conn = None
        if save_to_db:
            try:
                conn = open_db(db_path)
            except Exception as e:
                st.warning(f"Could not open the SQLite DB ({e}); results won't be saved.")
        overall = st.progress(0.0, text="Starting…")
        status = st.status(f"Running {len(lookups)} quer{'y' if len(lookups) == 1 else 'ies'} × {passes} passes…",
                           expanded=True)
        for i, q in enumerate(lookups, start=1):
            overall.progress((i - 1) / len(lookups), text=f"[{i}/{len(lookups)}] {q}")
            status.write(f"**{q}**")
            try:
                res = process_seed(q, cfg, client, status)
            except Exception as e:  # never lose the batch over one seed
                status.write(f"❌ {type(e).__name__}: {e}")
                continue
            if conn is not None:
                try:
                    res["run_id"] = save_result(conn, res, input_mode)
                except Exception as e:
                    status.write(f"⚠️ Could not save to SQLite: {e}")
            st.session_state.results.append(res)
            ok = sum(1 for p in res["passes"] if not p["error"])
            status.write(f"→ {len(res['queries_df'])} unique queries from {ok}/{passes} passes, "
                         f"{res['cluster_count']} clusters, {res['engine_requests']} requests.")
        overall.progress(1.0, text="Done")
        status.update(label="Complete.", state="complete", expanded=False)
        if conn is not None:
            conn.close()

# ---------------------------------------------------------------- views
tab_results, tab_history, tab_about = st.tabs(["Results", "History", "How it works"])

with tab_results:
    results = st.session_state.results
    if not results:
        st.info("Configure the run in the sidebar and press **Run Fan-Out 🚀**. "
                "Past runs are under **History**.")
    else:
        if len(results) > 1:
            summary = pd.DataFrame([{
                "seed query": r["seed"], "unique fan-outs": len(r["queries_df"]),
                "consensus": int((r["queries_df"]["weight"] >= 0.5).sum()) if not r["queries_df"].empty else 0,
                "clusters": r["cluster_count"],
                "passes ok": sum(1 for p in r["passes"] if not p["error"]),
                "requests": r["engine_requests"], "status": r["status"], "run id": r.get("run_id"),
                "error": next((p["error"] for p in r["passes"] if p["error"]), "") if r["queries_df"].empty else "",
            } for r in results])
            st.dataframe(summary, hide_index=True)
            frames = [export_frame(r) for r in results if not r["queries_df"].empty]
            if frames:
                all_csv = pd.concat(frames, ignore_index=True)
                st.download_button("Download all seeds (CSV)", all_csv.to_csv(index=False).encode("utf-8"),
                                   file_name="qforia_bulk.csv", mime="text/csv")
            else:
                st.error("None of the seed queries returned fan-out queries. "
                         "Check the error column above (usually the API key, billing, or rate limits).")
            idx = st.selectbox("Seed query", range(len(results)), format_func=lambda i: results[i]["seed"])
            st.markdown("---")
        else:
            idx = 0
        render_result(results[idx], key=f"res{idx}_{results[idx].get('run_id', 'x')}")

with tab_history:
    try:
        hconn = open_db(db_path)
        runs = list_saved_runs(hconn)
    except Exception as e:
        hconn, runs = None, pd.DataFrame()
        st.error(f"Could not open {db_path}: {e}")
    if hconn is not None and runs.empty:
        st.info("No v2 runs saved yet. (Runs from the previous version stay in the legacy `queries` table.)")
    elif hconn is not None:
        label = lambda rid: (lambda r: f"#{r.id} · {r.started_at} · {r.lookup_query} "
                                       f"({r.unique_queries} fan-outs, {r.passes} passes)")(runs[runs.id == rid].iloc[0])
        pick = st.selectbox("Open a saved run", runs["id"].tolist(), format_func=label)
        with st.expander("All saved runs", expanded=False):
            st.dataframe(runs, hide_index=True)
        saved = load_saved_run(hconn, int(pick))
        if saved:
            st.markdown("---")
            render_result(saved, key=f"hist{pick}")
    if hconn is not None:
        st.markdown("---")
        cexp1, cexp2 = st.columns(2)
        with cexp1:
            if st.button("Prepare export of all saved fan-out queries"):
                all_q = pd.read_sql_query(
                    """SELECT q.run_id, r.started_at, r.model_name, r.mode, r.passes, q.lookup_query, q.query,
                              q.qtype, q.user_intent, q.reasoning, q.routing_format, q.format_reason, q.frequency,
                              q.weight, q.avg_rank, q.passes_seen, q.cluster_id, q.cluster_label, q.position,
                              q.result_url
                         FROM fanout_queries q JOIN runs r ON r.id = q.run_id ORDER BY q.run_id DESC, q.frequency DESC""",
                    hconn)
                st.download_button(f"Download {len(all_q):,} rows (CSV)", all_q.to_csv(index=False).encode("utf-8"),
                                   file_name="qforia_fanout_queries_all.csv", mime="text/csv")
        with cexp2:
            if st.button("Prepare export of legacy (v1) queries"):
                legacy = pd.read_sql_query("SELECT * FROM queries ORDER BY id DESC", hconn)
                st.download_button(f"Download {len(legacy):,} legacy rows (CSV)",
                                   legacy.to_csv(index=False).encode("utf-8"),
                                   file_name="qforia_queries_legacy.csv", mime="text/csv")
        hconn.close()

with tab_about:
    st.markdown(f"""
**Qforia v{APP_VERSION}** simulates the query fan-out that AI Overviews / AI Mode perform before retrieval.

1. **Fan-out passes.** The same prompt is sent to Gemini *N* times in parallel (default `gemini-3.8-flash`,
   schema-enforced JSON via the Interactions API). Gemini 3.x runs at its default temperature, so passes differ the
   way repeated real-world fan-outs do.
2. **Counts.** Queries are matched across passes by exact text after lowercasing and stripping punctuation and extra
   spaces. **Freq** = number of passes that produced the query; **Weight** = Freq ÷ successful passes. High-weight
   queries are the stable core of the fan-out; 1x queries are the long tail.
3. **Diagram.** Seed → cluster (or type) → queries, with `Nx` badges. Edge thickness follows weight.
4. **Clusters{" (currently switched off)" if not CLUSTERING_ENABLED else ""}.** Unique queries are embedded with
   **EmbeddingGemma 2** (`{EMBED_MODEL_ID}`, `Clustering` task prompt), mean-centered, clustered with KMeans
   (k picked by cosine silhouette, max {MAX_CLUSTERS}), projected to 2-D (UMAP / t-SNE / PCA) and optionally named
   by Gemini. Turn it on with `CLUSTERING_ENABLED = True` in `qforia.py`.
5. **Engine requests** counts every Gemini call, including retries, fallbacks and cluster naming.

Runs are saved to SQLite (`fanout_passes`, `fanout_queries`, `fanout_clusters`, plus new columns on `runs`);
the v1 tables are left untouched.
""")
