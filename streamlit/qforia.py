import streamlit as st
import google.generativeai as genai
import pandas as pd
import json

# App config
st.set_page_config(page_title="Qforia", layout="wide")
st.title("🔍 Qforia: Query Fan-Out Simulator for AI Surfaces")

# Sidebar: API key input and query
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

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

# Configure Gemini (use 2.5 Pro)
if gemini_key:
    genai.configure(api_key=gemini_key)
    # You can change to a pinned version like "gemini-2.5-pro-exp-0827" if desired.
    model_name = "gemini-2.5-pro"
    model = genai.GenerativeModel(model_name)
else:
    st.error("Please enter your Gemini API Key to proceed.")
    st.stop()

# Allowed routing formats (sent to the model)
ALLOWED_FORMATS = [
    "web_article",
    "faq_page",
    "how_to_steps",
    "comparison_table",
    "buyers_guide",
    "checklist",
    "product_spec_sheet",
    "glossary/definition",
    "pricing_page",
    "review_roundup",
    "tutorial_video/transcript",
    "podcast_transcript",
    "code_samples/docs",
    "api_reference",
    "calculator/tool",
    "dataset",
    "image_gallery",
    "map/local_pack",
    "forum/qna",
    "pdf_whitepaper",
    "case_study",
    "press_release",
    "interactive_widget"
]

# Prompt builder
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
        "for retrieval and synthesis (e.g., a how-to should route to 'how_to_steps' or a video transcript; comparisons to 'comparison_table' or 'buyers_guide'). "
        "Choose exactly ONE label from this fixed list:\n"
        + ", ".join(ALLOWED_FORMATS) +
        ".\nReturn it in a field named 'routing_format' and give a short 'format_reason' (1 sentence)."
    )

    return (
        f"You are simulating Google's AI Mode query fan-out for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        f"Your first task is to determine the total number of queries to generate and the reasoning for this number:\n"
        f"{num_queries_instruction}\n\n"
        f"Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.\n"
        f"Each of the following transformation types MUST be represented at least once, if the total allows:\n"
        f"1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
        f"The 'reasoning' field for each query should explain why that query was generated (tie it to the original query, its type, and user intent). "
        f"Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        f"{routing_note}\n\n"
        f"Return only a valid JSON object in this exact schema:\n"
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

# Single fan-out
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

# Initialize session state
if 'last_runs' not in st.session_state:
    st.session_state.last_runs = []

# Run button
if st.sidebar.button("Run Fan-Out 🚀"):
    # Build list of lookup queries
    if input_mode == "Single query":
        lookups = [user_query.strip()] if user_query.strip() else []
    else:
        lookups = [q.strip() for q in bulk_text.splitlines() if q.strip()]

    if not lookups:
        st.warning("⚠️ Please provide at least one query.")
        st.stop()

    all_rows = []
    run_summaries = []
    errors = []

    status = st.status("Processing queries…", expanded=True)
    progress = st.progress(0)
    total = len(lookups)

    for i, q in enumerate(lookups, start=1):
        try:
            details, expanded, raw = generate_fanout(q, mode)
            run_summaries.append({
                "lookup_query": q,
                "target_query_count": details.get("target_query_count"),
                "reasoning_for_count": details.get("reasoning_for_count", "")
            })
            # Flatten rows, prefix with lookup query
            for obj in expanded:
                all_rows.append({
                    "lookup_query": q,
                    "query": obj.get("query", ""),
                    "type": obj.get("type", ""),
                    "user_intent": obj.get("user_intent", ""),
                    "reasoning": obj.get("reasoning", ""),
                    "routing_format": obj.get("routing_format", ""),
                    "format_reason": obj.get("format_reason", "")
                })
            status.write(f"✅ Processed: **{q}** — generated {len(expanded)} queries.")
        except json.JSONDecodeError as e:
            msg = f"❌ JSON parse failed for '{q}': {e}"
            status.write(msg)
            errors.append({"lookup_query": q, "error": str(e)})
        except Exception as e:
            msg = f"❌ Error for '{q}': {e}"
            status.write(msg)
            errors.append({"lookup_query": q, "error": str(e)})

        progress.progress(i / total)

    status.update(label="Complete.", state="complete")

    # Build output DataFrame (lookup_query first)
    if all_rows:
        df = pd.DataFrame(all_rows)

        # Ensure column order (lookup_query first)
        preferred_cols = [
            "lookup_query",
            "query",
            "type",
            "user_intent",
            "reasoning",
            "routing_format",
            "format_reason"
        ]
        existing = [c for c in preferred_cols if c in df.columns]
        others = [c for c in df.columns if c not in existing]
        df = df[existing + others]

        st.subheader("📊 Synthetic Queries (with routing format)")
        st.dataframe(df, use_container_width=True, height=(min(len(df), 20) + 1) * 35 + 3)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="qforia_output_bulk_with_routing.csv", mime="text/csv")
    else:
        st.warning("No synthetic queries were generated.")

    # Summaries per lookup (optional)
    if run_summaries:
        st.markdown("---")
        st.subheader("🧠 Generation Plans (per lookup)")
        sum_df = pd.DataFrame(run_summaries)
        st.dataframe(sum_df, use_container_width=True)

    # Error table if any
    if errors:
        st.markdown("---")
        st.subheader("⚠️ Errors")
        err_df = pd.DataFrame(errors)
        st.dataframe(err_df, use_container_width=True)
