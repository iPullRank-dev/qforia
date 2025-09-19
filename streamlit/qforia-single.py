import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re

# App config
st.set_page_config(page_title="Qforia", layout="wide")
st.title("🔍 Qforia: Query Fan-Out Simulator for AI Surfaces")

# Sidebar: API key input and query
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
user_query = st.sidebar.text_area("Enter your query", "What's the best electric SUV for driving up mt rainier?", height=120)
mode = st.sidebar.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    # Ensure you are using a model that supports longer/complex JSON outputs well.
    # The user's model "gemini-2.5-flash-preview-05-20" might be a specific version;
    # if issues arise, consider trying "gemini-1.5-flash-latest" or "gemini-1.5-pro-latest".
    model = genai.GenerativeModel("gemini-1.5-flash-latest") # Using a common recent flash model
else:
    st.error("Please enter your Gemini API Key to proceed.")
    st.stop()

# Prompt with detailed Chain-of-Thought logic
def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"**you must decide on an optimal number of queries to generate.** "
            f"This number must be **at least {min_queries_simple}**. "
            f"For a straightforward query, generating around {min_queries_simple}-{min_queries_simple + 2} queries might be sufficient. "
            f"If the query has a few distinct aspects or common follow-up questions, aim for a slightly higher number, perhaps {min_queries_simple + 3}-{min_queries_simple + 5} queries. "
            f"Provide a brief reasoning for why you chose this specific number of queries. The queries themselves should be tightly scoped and highly relevant."
        )
    else:  # AI Mode (complex)
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"**you must decide on an optimal number of queries to generate.** "
            f"This number must be **at least {min_queries_complex}**. "
            f"For multifaceted queries requiring exploration of various angles, sub-topics, comparisons, or deeper implications, "
            f"you should generate a more comprehensive set, potentially {min_queries_complex + 5}-{min_queries_complex + 10} queries, or even more if the query is exceptionally broad or deep. "
            f"Provide a brief reasoning for why you chose this specific number of queries. The queries should be diverse and in-depth."
        )

    return (
        f"You are simulating Google's AI Mode query fan-out process for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        f"**Your first task is to determine the total number of queries to generate and the reasoning for this number, based on the instructions below:**\n"
        f"{num_queries_instruction}\n\n"
        f"**Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.**\n"
        "Each of the following query transformation types MUST be represented at least once in the generated set, if the total number of queries you decide to generate allows for it (e.g., if you generate 12 queries, try to include all 6 types at least once, and then add more of the relevant types):\n"
        "1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
        "The 'reasoning' field for each *individual query* should explain why that specific query was generated in relation to the original query, its type, and the overall user intent.\n"
        "Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        "Return only a valid JSON object. The JSON object should strictly follow this format:\n"
        "{\n"
        "  \"generation_details\": {\n"
        "    \"target_query_count\": 12, // This is an EXAMPLE number; you will DETERMINE the actual number based on your analysis.\n"
        "    \"reasoning_for_count\": \"The user query was moderately complex, so I chose to generate slightly more than the minimum for a simple overview to cover key aspects like X, Y, and Z.\" // This is an EXAMPLE reasoning; provide your own.\n"
        "  },\n"
        "  \"expanded_queries\": [\n"
        "    // Array of query objects. The length of this array MUST match your 'target_query_count'.\n"
        "    {\n"
        "      \"query\": \"Example query 1...\",\n"
        "      \"type\": \"reformulation\",\n"
        "      \"user_intent\": \"Example intent...\",\n"
        "      \"reasoning\": \"Example reasoning for this specific query...\"\n"
        "    },\n"
        "    // ... more query objects ...\n"
        "  ]\n"
        "}"
    )

# Fan-out generation function
def generate_fanout(query, mode):
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean potential markdown code block fences
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        data = json.loads(json_text)
        generation_details = data.get("generation_details", {})
        expanded_queries = data.get("expanded_queries", [])

        # Store details for display
        st.session_state.generation_details = generation_details

        return expanded_queries
    except json.JSONDecodeError as e:
        st.error(f"🔴 Failed to parse Gemini response as JSON: {e}")
        st.text("Raw response that caused error:")
        st.text(json_text if 'json_text' in locals() else "N/A (error before json_text assignment)")
        st.session_state.generation_details = None
        return None
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during generation: {e}")
        if hasattr(response, 'text'):
             st.text("Raw response content (if available):")
             st.text(response.text)
        st.session_state.generation_details = None
        return None

# Initialize session state for generation_details if not present
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None

# Generate and display results
if st.sidebar.button("Run Fan-Out 🚀"):
    # Clear previous details
    st.session_state.generation_details = None
    
    if not user_query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("🤖 Generating query fan-out using Gemini... This may take a moment..."):
            results = generate_fanout(user_query, mode)

        if results: # Check if results is not None and not empty
            st.success("✅ Query fan-out complete!")

            # Display the reasoning for the count if available
            if st.session_state.generation_details:
                details = st.session_state.generation_details
                generated_count = len(results)
                target_count_model = details.get('target_query_count', 'N/A')
                reasoning_model = details.get('reasoning_for_count', 'Not provided by model.')

                st.markdown("---")
                st.subheader("🧠 Model's Query Generation Plan")
                st.markdown(f"🔹 **Target Number of Queries Decided by Model:** `{target_count_model}`")
                st.markdown(f"🔹 **Model's Reasoning for This Number:** _{reasoning_model}_")
                st.markdown(f"🔹 **Actual Number of Queries Generated:** `{generated_count}`")
                st.markdown("---")
                
                if isinstance(target_count_model, int) and target_count_model != generated_count:
                    st.warning(f"⚠️ Note: Model aimed to generate {target_count_model} queries but actually produced {generated_count}.")
            else:
                 st.info("ℹ️ Generation details (target count, reasoning) were not available from the model's response.")


            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, height=(min(len(df), 20) + 1) * 35 + 3) # Dynamic height

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv, file_name="qforia_output.csv", mime="text/csv")
        
        elif results is None: # Error occurred in generate_fanout
            # Error message is already displayed by generate_fanout
            pass
        else: # Handle empty results list (empty list, not None)
            st.warning("⚠️ No queries were generated. The model returned an empty list, or there was an issue.")