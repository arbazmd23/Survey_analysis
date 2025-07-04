import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any
from collections import Counter
import anthropic

# ───────────────────────────────
# Page Configuration
# ───────────────────────────────
st.set_page_config(
    page_title="Survey Analysis with AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────
# Initialize Anthropic Client
# ───────────────────────────────
@st.cache_resource
def get_anthropic_client():
    """Initialize Anthropic client using Streamlit secrets"""
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Error loading Anthropic API key: {e}")
        st.stop()

client = get_anthropic_client()

# ───────────────────────────────
# Utilities (Same as original)
# ───────────────────────────────
def extract_all_responses(json_data: List[dict]) -> List[Dict[str, Any]]:
    all_responses = []
    for respondent in json_data:
        rid = respondent.get("respondent_id", "unknown")
        responses = respondent.get("responses", [])

        role = next((r.get("response") for r in responses if "role" in r.get("question", "").lower()), "Unknown")
        industry = next((r.get("response") for r in responses if "industry" in r.get("question", "").lower() or "domain" in r.get("question", "").lower()), "Unknown")

        for r in responses:
            response_text = str(r.get("response", "")).strip()
            question_text = r.get("question", "").strip()
            if len(response_text) >= 1:
                all_responses.append({
                    "respondent_id": rid,
                    "question": question_text,
                    "text": response_text,
                    "role": role,
                    "industry": industry
                })
    return all_responses

def generate_limited_word_cloud(responses: List[Dict[str, Any]]) -> Dict[str, int]:
    text_content = " ".join([resp["text"] for resp in responses]).lower()
    words = re.findall(r"\\b[a-zA-Z]{3,}\\b", text_content)
    stop_words = set([
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her',
        'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man',
        'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let',
        'put', 'say', 'she', 'too', 'use', 'yes', 'no', 'very', 'much'
    ])
    filtered = [w for w in words if w not in stop_words]
    freq = Counter(filtered)
    return dict(freq.most_common(5))

def analyze_with_claude(responses: List[Dict[str, Any]], word_data: Dict[str, int]) -> Dict[str, Any]:
    survey_context = [
        {
            "question": r["question"],
            "response": r["text"],
            "role": r["role"],
            "industry": r["industry"]
        } for r in responses
    ]

    prompt = f"""
You are an expert startup advisor analyzing customer survey data.

SURVEY RESPONSES:
{json.dumps(survey_context, indent=2)}

TOP THEMES:
{json.dumps(word_data)}

TASK:
Analyze the responses and return only valid JSON:
{{
  "clusters": [{{"cluster_id": "1", "title": "...", "customer_segment": "...", "pain_intensity": 4, "key_insight": "...", "recommended_action": "...", "supporting_quotes": [...], "sentiment": "...", "urgency": 4, "response_count": 2}}],
  "sentiment_analysis": {{"overall_sentiment": {{"positive": 0, "negative": 0, "neutral": 0}}, "average_pain_level": 3, "key_emotions": ["..."]}},
  "executive_summary": {{"market_opportunity": "...", "primary_segment": "...", "immediate_actions": ["..."], "risk_factors": ["..."], "confidence_level": 4}},
  "wordCloudData": {json.dumps(word_data)}
}}
    """

    try:
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
            "total_responses": len(responses),
            "model_used": "claude-3-5-sonnet-20241022",
            "analysis_method": "AI-powered clustering and analysis"
        }
        return result

    except Exception as e:
        raise Exception(f"AI analysis failed: {e}")

# ───────────────────────────────
# Streamlit UI
# ───────────────────────────────
def main():
    st.title("📊 Survey Analysis with AI")
    st.markdown("Upload your survey data as JSON and get AI-powered insights using Claude")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Prepare your JSON data** in the format shown in the example
        2. **Paste the JSON** in the text area below
        3. **Click 'Analyze Survey'** to get insights
        4. **Review the JSON results** 
        """)
        
        st.header("📄 JSON Format Example")
        example_json = [
            {
                "respondent_id": "001",
                "responses": [
                    {
                        "question": "What is your role?",
                        "response": "Product Manager"
                    },
                    {
                        "question": "What industry do you work in?",
                        "response": "Technology"
                    },
                    {
                        "question": "What challenges do you face?",
                        "response": "Managing multiple projects simultaneously"
                    }
                ]
            }
        ]
        st.json(example_json)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 JSON Input")
        json_input = st.text_area(
            "Paste your survey JSON data here:",
            height=300,
            placeholder="Paste your JSON data here...",
            help="Make sure your JSON follows the format shown in the sidebar example"
        )
    
    with col2:
        st.header("⚙️ Analysis Options")
        
        # Health check
        if st.button("🔍 Test Connection"):
            try:
                # Simple test to verify the client works
                test_response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                st.success("✅ Connection to Claude API successful!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
        
        # Analyze button
        analyze_button = st.button("🚀 Analyze Survey", type="primary")

    # Analysis section
    if analyze_button:
        if not json_input.strip():
            st.warning("⚠️ Please paste your JSON data first.")
            return
        
        try:
            # Parse JSON
            with st.spinner("🔄 Parsing JSON data..."):
                survey_data = json.loads(json_input)
            
            # Extract responses
            with st.spinner("🔄 Extracting responses..."):
                all_responses = extract_all_responses(survey_data)
            
            if not all_responses:
                st.error("❌ No valid survey responses found in the JSON data.")
                return
            
            st.success(f"✅ Found {len(all_responses)} responses from {len(survey_data)} respondents")
            
            # Generate word cloud
            with st.spinner("🔄 Generating word analysis..."):
                word_cloud = generate_limited_word_cloud(all_responses)
            
            # Analyze with Claude
            with st.spinner("🔄 Analyzing with Claude AI... This may take a moment..."):
                analysis_result = analyze_with_claude(all_responses, word_cloud)
            
            # Display results as JSON only
            st.header("📊 Analysis Results")
            
            # Show basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Responses", len(all_responses))
            with col2:
                st.metric("Total Respondents", len(survey_data))
            with col3:
                st.metric("Analysis Complete", "✅")
            
            # Display the JSON output
            st.subheader("🔍 Analysis JSON Output")
            st.json(analysis_result)
            
            # Option to download JSON
            json_str = json.dumps(analysis_result, indent=2)
            st.download_button(
                label="📥 Download JSON Results",
                data=json_str,
                file_name="survey_analysis_results.json",
                mime="application/json"
            )
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {e}")
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()