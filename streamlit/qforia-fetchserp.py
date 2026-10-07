import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import sqlite3
from datetime import datetime
import time
import requests
from urllib.parse import urlparse

# =========================
# App config
# =========================
st.set_page_config(page_title="Qforia", layout="wide")
st.title("🔍 Qforia: Query Fan-Out Simulator for AI Surfaces")

# =========================
# Sidebar: config
# =========================
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

# SQLite DB path
db_path = st.sidebar.text_input("SQLite DB path", value="qforia.sqlite")

input_mode = st.sidebar.radio("Input Mode", ["Single query", "Bulk list"])
if input_mode == "Single query":
    user_query = st.sidebar.text_area(
        "Enter your query",
        "What's the best electric SUV for driving up mt rainier?",
        height=120
    )
else:
    bulk_text = st.sidebar.text_area(
        "Paste queries (one per line)",
        "best electric suv for snow\nsleep training methods for toddlers\nhow to freeze sourdough starter",
        height=180
    )

mode = st.sidebar.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Optional: Rank Lookups (FetchSERP /api/v1/ranking)")
enable_serp = st.sidebar.checkbox("Enable FetchSERP rank lookups", value=False)
fetchserp_key = st.sidebar.text_input("FetchSERP API Key", type="password", disabled=not enable_serp)
target_domain = st.sidebar.text_input("Domain to match (e.g., example.com)", disabled=not enable_serp)
serp_engine = st.sidebar.selectbox("Search engine", ["google", "bing", "yahoo", "duckduckgo"], disabled=not enable_serp)
serp_country = st.sidebar.text_input("Country (2-letter, e.g., us, uk, fr)", value="us", disabled=not enable_serp)
pages_number = st.sidebar.number_input("Pages to scan (1-30)", min_value=1, max_value=30, value=1, step=1, disabled=not enable_serp)
serp_delay = st.sidebar.number_input("Delay between ranking calls (seconds)", min_value=0.0, max_value=10.0, value=0.2, step=0.1, disabled=not enable_serp)

# =========================
# Gemini model (2.5 Pro exact string)
# =========================
if gemini_key:
    genai.configure(api_key=gemini_key)
    model_name = "gemini-2.5-pro"  # per your requirement
    model = genai.GenerativeModel(model_name)
else:
    st.error("Please enter your Gemini API Key to proceed.")
    st.stop()

# =========================
# Routing formats
# =========================
ALLOWED_FORMATS = [
    "web_article","faq_page","how_to_steps","comparison_table","buyers_guide","checklist",
    "product_spec_sheet","glossary/definition","pricing_page","review_roundup",
    "tutorial_video/transcript","podcast_transcript","code_samples/docs","api_reference",
    "calculator/tool","dataset","image_gallery","map/local_pack","forum/qna",
    "pdf_whitepaper","case_study","press_release","interactive_widget"
]

# =========================
# SQLite helpers
# =========================
def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
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
        )
    """)
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
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            created_at TEXT NOT NULL,
            lookup_query TEXT NOT NULL,
            error TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)
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
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_queries_lookup ON queries(lookup_query)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_queries_run ON queries(run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_serp_run ON serp_results(run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_serp_lookup ON serp_results(lookup_query)")
    conn.commit()

def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    init_db(conn)
    return conn

def insert_run(conn, mode, model_name, input_mode, lookup_query, gen_details):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs (started_at, mode, model_name, input_mode, lookup_query, target_query_count, reasoning_for_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        mode, model_name, input_mode, lookup_query,
        gen_details.get("target_query_count"),
        gen_details.get("reasoning_for_count")
    ))
    conn.commit()
    return cur.lastrowid

def insert_queries(conn, run_id, lookup_query, rows):
    cur = conn.cursor()
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    to_insert = []
    for r in rows:
        to_insert.append((
            run_id, created_at, lookup_query,
            r.get("query",""),
            r.get("type",""),
            r.get("user_intent",""),
            r.get("reasoning",""),
            r.get("routing_format",""),
            r.get("format_reason","")
        ))
    cur.executemany("""
        INSERT OR IGNORE INTO queries
        (run_id, created_at, lookup_query, query, qtype, user_intent, reasoning, routing_format, format_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, to_insert)
    conn.commit()
    # Map (lookup_query, query) -> query_id
    m = {}
    for r in rows:
        qtxt = r.get("query","")
        row = cur.execute("""
          SELECT id FROM queries WHERE run_id=? AND lookup_query=? AND query=? ORDER BY id DESC LIMIT 1
        """, (run_id, lookup_query, qtxt)).fetchone()
        if row:
            m[(lookup_query, qtxt)] = row[0]
    return m

def insert_error(conn, run_id, lookup_query, error_msg):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO errors (run_id, created_at, lookup_query, error)
        VALUES (?, ?, ?, ?)
    """, (
        run_id,
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        lookup_query,
        str(error_msg)
    ))
    conn.commit()

def insert_serp_result(conn, run_id, query_id, lookup_query, synthetic_query, meta, hit, raw_json):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO serp_results
        (run_id, query_id, created_at, lookup_query, synthetic_query, search_engine, country, pages_number,
         matched_domain, position, result_url, result_title, site_name, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, query_id,
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        lookup_query, synthetic_query,
        meta.get("engine"), meta.get("country"), meta.get("pages"),
        meta.get("domain"), hit.get("ranking"), hit.get("url"), hit.get("title"), hit.get("site_name"),
        json.dumps(raw_json, ensure_ascii=False)[:1_000_000]
    ))
    conn.commit()

# =========================
# Prompt builder
# =========================
def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"you must decide on an optimal number of queries to generate. "
            f"This number must be at least {min_queries_simple}. "
            f"For a straightforward query, generate around {min_queries_simple}-{min_queries_simple + 2}. "
            f"If the query has a few distinct aspects or common follow-ups, aim for {min_queries_simple + 3}-{min_queries_simple + 5}. "
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
        "For EACH expanded query, also identify the most likely CONTENT TYPE / FORMAT the routing system would prefer "
        "for retrieval and synthesis. Choose exactly ONE label from this fixed list:\n"
        + ", ".join(ALLOWED_FORMATS) +
        ". Return it in 'routing_format' and give a short 'format_reason' (1 sentence)."
    )

    return (
        f"You are simulating Google's AI Mode query fan-out for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        f"Your first task is to determine the total number of queries to generate and the reasoning for this number:\n"
        f"{num_queries_instruction}\n\n"
        f"Once decided, generate exactly that many unique synthetic queries.\n"
        f"Each of these transformation types MUST appear at least once, if totals allow:\n"
        f"1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
        f"The 'reasoning' for each query should tie back to the original query, type, and user intent. "
        f"Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        f"{routing_note}\n\n"
        f"Return only a valid JSON object in this schema:\n"
        "{\n"
        "  \"generation_details\": {\n"
        "    \"target_query_count\": 12,\n"
        "    \"reasoning_for_count\": \"...\"\n"
        "  },\n"
        "  \"expanded_queries\": [\n"
        "    {\n"
        "      \"query\": \"...\",\n"
        "      \"type\": \"reformulation | related | implicit | comparative | entity_expansion | personalized\",\n"
        "      \"user_intent\": \"...\",\n"
        "      \"reasoning\": \"...\",\n"
        "      \"routing_format\": \"one_of_allowed_labels\",\n"
        "      \"format_reason\": \"one sentence why this format is best\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

# =========================
# Generation
# =========================
def generate_fanout(query, mode):
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    response = model.generate_content(prompt)
    json_text = response.text.strip()

    # Clean code fences if present
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    json_text = json_text.strip()

    data = json.loads(json_text)
    generation_details = data.get("generation_details", {})
    expanded_queries = data.get("expanded_queries", [])
    return generation_details, expanded_queries, json_text

# =========================
# FetchSERP ranking helpers
# =========================
FETCHSERP_RANKING = "https://www.fetchserp.com/api/v1/ranking"

def normalize_domain(d: str) -> str:
    d = d.strip().lower()
    if d.startswith("http://") or d.startswith("https://"):
        d = urlparse(d).netloc
    if d.startswith("www."):
        d = d[4:]
    return d

def fetchserp_ranking_lookup(query_text: str, api_key: str, engine: str, country: str, domain: str, pages: int):
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": query_text,
        "search_engine": engine,
        "country": country,
        "domain": domain,
        "pages_number": pages
    }
    r = requests.get(FETCHSERP_RANKING, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# =========================
# Session state
# =========================
if 'last_runs' not in st.session_state:
    st.session_state.last_runs = []

# =========================
# Run
# =========================
if st.sidebar.button("Run Fan-Out 🚀"):
    # Build lookup list
    if input_mode == "Single query":
        lookups = [user_query.strip()] if user_query.strip() else []
    else:
        lookups = [q.strip() for q in bulk_text.splitlines() if q.strip()]

    if not lookups:
        st.warning("⚠️ Please provide at least one query.")
        st.stop()

    if enable_serp:
        if not fetchserp_key:
            st.error("FetchSERP lookups enabled but no API key provided.")
            st.stop()
        if not target_domain:
            st.error("Please provide a domain to match (e.g., example.com).")
            st.stop()
        matched_domain_norm = normalize_domain(target_domain)

    # Open DB
    conn = open_db(db_path)

    all_rows = []
    run_summaries = []
    errors = []

    status = st.status("Processing queries…", expanded=True)
    progress = st.progress(0)
    total = len(lookups)

    for i, q in enumerate(lookups, start=1):
        # Pre-create a run row
        run_id = insert_run(
            conn,
            mode=mode,
            model_name=model_name,
            input_mode=input_mode,
            lookup_query=q,
            gen_details={"target_query_count": None, "reasoning_for_count": None}
        )

        try:
            details, expanded, raw = generate_fanout(q, mode)

            # Update run row with details
            conn.execute("""
                UPDATE runs
                   SET target_query_count = ?, reasoning_for_count = ?
                 WHERE id = ?
            """, (details.get("target_query_count"), details.get("reasoning_for_count"), run_id))
            conn.commit()

            # Persist queries and get IDs
            query_id_map = insert_queries(conn, run_id, q, expanded)

            # In-memory output table
            for obj in expanded:
                all_rows.append({
                    "lookup_query": q,
                    "query": obj.get("query", ""),
                    "type": obj.get("type", ""),
                    "user_intent": obj.get("user_intent", ""),
                    "reasoning": obj.get("reasoning", ""),
                    "routing_format": obj.get("routing_format", ""),
                    "format_reason": obj.get("format_reason", ""),
                    # placeholders for serp columns
                    "position": None,
                    "result_url": None,
                    "result_title": None
                })

            # Optional ranking lookups (cheaper)
            if enable_serp and expanded:
                for obj in expanded:
                    sq = obj.get("query", "")
                    try:
                        data = fetchserp_ranking_lookup(
                            sq, fetchserp_key, serp_engine, serp_country,
                            matched_domain_norm, pages_number
                        )
                        ddata = data.get("data", {})
                        hit = {
                            "ranking": ddata.get("ranking"),
                            "url": ddata.get("url"),
                            "title": ddata.get("title"),
                            "site_name": ddata.get("site_name")
                        }

                        qid = query_id_map.get((q, sq))
                        meta = {
                            "engine": serp_engine,
                            "country": serp_country,
                            "pages": pages_number,
                            "domain": matched_domain_norm
                        }

                        insert_serp_result(conn, run_id, qid, q, sq, meta, hit, ddata)

                        # Update in-memory row
                        for row in all_rows:
                            if row["lookup_query"] == q and row["query"] == sq:
                                row["position"] = hit.get("ranking")
                                row["result_url"] = hit.get("url")
                                row["result_title"] = hit.get("title")
                                break

                        time.sleep(float(serp_delay))
                    except Exception as e:
                        insert_error(conn, run_id, q, f"FetchSERP ranking error for '{sq}': {e}")

            run_summaries.append({
                "lookup_query": q,
                "target_query_count": details.get("target_query_count"),
                "reasoning_for_count": details.get("reasoning_for_count", "")
            })

            status.write(f"✅ Processed: **{q}** — {len(expanded)} queries{' + ranking lookups' if enable_serp else ''} (run_id={run_id}).")

        except json.JSONDecodeError as e:
            insert_error(conn, run_id, q, f"JSON parse failed: {e}")
            status.write(f"❌ JSON parse failed for '{q}': {e}")
            errors.append({"lookup_query": q, "error": str(e), "run_id": run_id})
        except Exception as e:
            insert_error(conn, run_id, q, f"Unexpected error: {e}")
            status.write(f"❌ Error for '{q}': {e}")
            errors.append({"lookup_query": q, "error": str(e), "run_id": run_id})

        progress.progress(i / total)

    status.update(label="Complete.", state="complete")

    # =========================
    # On-screen results + CSV
    # =========================
    if all_rows:
        df = pd.DataFrame(all_rows)
        preferred_cols = [
            "lookup_query","query","type","user_intent","reasoning","routing_format","format_reason",
            "position","result_url","result_title"
        ]
        existing = [c for c in df.columns if c in preferred_cols]
        others = [c for c in df.columns if c not in preferred_cols]
        df = df[preferred_cols] if not others else df[preferred_cols + others]

        st.subheader("📊 Synthetic Queries" + (" + Ranking Positions" if enable_serp else ""))
        st.dataframe(df, use_container_width=True, height=(min(len(df), 20) + 1) * 35 + 3)

        csv = df.to_csv(index=False).encode("utf-8")
        fname = "qforia_output_with_ranking.csv" if enable_serp else "qforia_output.csv"
        st.download_button("📥 Download CSV (this run)", data=csv, file_name=fname, mime="text/csv")
    else:
        st.warning("No synthetic queries were generated.")

    if run_summaries:
        st.markdown("---")
        st.subheader("🧠 Generation Plans (per lookup)")
        sum_df = pd.DataFrame(run_summaries)
        st.dataframe(sum_df, use_container_width=True)

    if errors:
        st.markdown("---")
        st.subheader("⚠️ Errors (also stored in DB)")
        err_df = pd.DataFrame(errors)
        st.dataframe(err_df, use_container_width=True)

    conn.close()

# =========================
# Quick DB viewer/export
# =========================
st.markdown("---")
st.subheader("📚 SQLite Browser")
col1, col2, col3 = st.columns([2,1,1])

with col1:
    view_limit = st.number_input("Rows to preview from 'queries' table", min_value=10, max_value=10000, value=200, step=10)
with col2:
    do_preview = st.button("Preview stored queries")
with col3:
    do_export = st.button("Export all stored queries (CSV)")

if do_preview or do_export:
    try:
        conn = open_db(db_path)
        qdf = pd.read_sql_query(
            f"SELECT id, run_id, created_at, lookup_query, query, qtype, user_intent, reasoning, routing_format, format_reason FROM queries ORDER BY id DESC {'LIMIT ' + str(view_limit) if do_preview else ''};",
            conn
        )
        if do_preview:
            st.dataframe(qdf, use_container_width=True, height=(min(len(qdf), 20) + 1) * 35 + 3)
        if do_export:
            csv_all = qdf.to_csv(index=False).encode("utf-8")
            st.download_button("📤 Download all stored queries (CSV)", data=csv_all, file_name="qforia_queries_all.csv", mime="text/csv")
        conn.close()
    except Exception as e:
        st.error(f"DB error: {e}")
