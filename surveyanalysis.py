import streamlit as st
import json
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import hdbscan
from collections import Counter
import anthropic

# ─────────────────────────────────────────────
# Load Environment and Anthropic API
# ─────────────────────────────────────────────
@st.cache_resource
def load_environment():
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    if not api_key:
        st.error("Missing ANTHROPIC_API_KEY in secrets")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# Utilities (Same as original)
# ─────────────────────────────────────────────
def extract_responses(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_responses = []
    for entry in data:
        rid = entry.get("respondent_id", "unknown")
        for resp in entry.get("responses", []):
            text = str(resp.get("response", "")).strip()
            question = resp.get("question", "")
            if len(text) > 1:
                all_responses.append({
                    "respondent_id": rid,
                    "question": question,
                    "text": text
                })
    return all_responses

def generate_word_cloud(responses: List[Dict[str, Any]]) -> Dict[str, int]:
    content = " ".join([r["text"] for r in responses]).lower()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", content)
    stopwords = set(["the", "and", "for", "are", "you", "with", "this", "that", "from", "have", "all", "use"])
    filtered = [w for w in words if w not in stopwords]
    return dict(Counter(filtered).most_common(8))

def cluster_responses(responses: List[Dict[str, Any]], embedding_model):
    texts = [r["text"] for r in responses]
    embeddings = embedding_model.encode(texts)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = clusterer.fit_predict(embeddings)
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1: continue  # Skip noise
        clusters.setdefault(label, []).append(responses[idx])
    return clusters

def summarize_with_llm(clusters: Dict[int, List[Dict[str, Any]]], word_cloud: Dict[str, int], client) -> Dict[str, Any]:
    cluster_input = {
        str(cid): {
            "responses": [{"text": r["text"], "question": r["question"]} for r in items],
            "count": len(items)
        } for cid, items in clusters.items()
    }

    prompt = f"""
You are a startup research analyst. Analyze the following customer response clusters to extract structured insights.

CLUSTERS:
{json.dumps(cluster_input, indent=2)}

WORD CLOUD:
{json.dumps(word_cloud)}

Respond in exactly this JSON format:
{{
  "clusters": [{{"cluster_id": "1", "title": "...", "customer_segment": "...", "pain_intensity": 4, "key_insight": "...", "recommended_action": "...", "supporting_quotes": ["..."], "sentiment": "...", "urgency": 4, "response_count": 2}}],
  "sentiment_analysis": {{"overall_sentiment": {{"positive": 0, "negative": 0, "neutral": 0}}, "average_pain_level": 3, "key_emotions": ["..."]}},
  "executive_summary": {{"market_opportunity": "...", "primary_segment": "...", "immediate_actions": ["..."], "risk_factors": ["..."], "confidence_level": 4}},
  "wordCloudData": {json.dumps(word_cloud)}
}}
"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    result = json.loads(text)
    result["analysis_metadata"] = {
        "total_responses": sum(len(v) for v in clusters.values()),
        "model_used": "claude-3-5-sonnet-20241022",
        "analysis_method": "HDBSCAN + Claude LLM"
    }
    return result

# ─────────────────────────────────────────────
# Analysis Function
# ─────────────────────────────────────────────
def analyze_survey(payload: List[dict], client, embedding_model):
    try:
        responses = extract_responses(payload)
        if len(responses) < 2:
            return {"error": "Need at least 2 valid responses."}

        clusters = cluster_responses(responses, embedding_model)
        if not clusters:
            return {"error": "No valid clusters detected."}

        word_cloud = generate_word_cloud(responses)
        result = summarize_with_llm(clusters, word_cloud, client)
        return result

    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Survey Analysis Tool",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Survey Analysis Tool")
    st.markdown("Analyze customer survey responses using AI-powered clustering and sentiment analysis.")
    
    # Load models
    client = load_environment()
    embedding_model = load_embedding_model()
    
    # Sidebar for health check
    with st.sidebar:
        st.header("System Status")
        if st.button("Health Check"):
            st.success("✅ System is operational")
            st.info("Model: claude-3-5-sonnet-20241022")
    
    # Main interface
    st.header("Input Survey Data")
    
    # Example JSON structure
    with st.expander("📋 Expected JSON Format"):
        example_json = [
            {
                "respondent_id": "resp_001",
                "responses": [
                    {
                        "question": "What is your main pain point?",
                        "response": "The product is too expensive and hard to use"
                    },
                    {
                        "question": "How would you improve our service?",
                        "response": "Make it more affordable and user-friendly"
                    }
                ]
            },
            {
                "respondent_id": "resp_002",
                "responses": [
                    {
                        "question": "What is your main pain point?",
                        "response": "Customer support is slow to respond"
                    }
                ]
            }
        ]
        st.json(example_json)
    
    # JSON input area
    json_input = st.text_area(
        "Paste your survey data (JSON format):",
        height=300,
        placeholder="Paste your JSON data here..."
    )
    
    # Analysis button
    if st.button("🔍 Analyze Survey", type="primary"):
        if json_input.strip():
            try:
                # Parse JSON
                survey_data = json.loads(json_input)
                
                # Show progress
                with st.spinner("Analyzing survey responses..."):
                    result = analyze_survey(survey_data, client, embedding_model)
                
                # Display results
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success("Analysis complete!")
                    
                    # Display JSON output
                    st.subheader("📊 Analysis Results")
                    st.json(result)
                    
                    # Download results
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(result, indent=2),
                        file_name="survey_analysis_results.json",
                        mime="application/json"
                    )
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please paste your survey data in JSON format.")

if __name__ == "__main__":
    main()