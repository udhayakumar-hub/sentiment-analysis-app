import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.analyzer import SentimentAnalyzer
from src.preprocessor import TextPreprocessor
from src.utils import SAMPLE_TEXTS, add_to_history, get_history_df, calculate_statistics

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis App | NLP Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2.2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    
    .main-header p {
        color: #C7D2FE;
        font-size: 1.05rem;
        max-width: 800px;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #EEF2FF;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }

    /* Result Cards */
    .result-card-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1.5px solid #10B981;
        border-radius: 16px;
        padding: 1.6rem;
        color: #064E3B;
    }
    
    .result-card-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 16px;
        padding: 1.6rem;
        color: #7F1D1D;
    }
    
    .result-card-neutral {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1.5px solid #F59E0B;
        border-radius: 16px;
        padding: 1.6rem;
        color: #78350F;
    }
    
    .sentiment-title {
        font-size: 1.85rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
    }
    
    .word-tag-pos {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
        border: 1px solid #A7F3D0;
    }
    
    .word-tag-neg {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
        border: 1px solid #FECACA;
    }

    .word-tag-token {
        background-color: #F1F5F9;
        color: #334155;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-family: monospace;
        display: inline-block;
        margin: 2px;
        border: 1px solid #E2E8F0;
    }
    
    .pipeline-step {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .pipeline-step h4 {
        margin: 0 0 0.4rem 0;
        color: #1E293B;
        font-size: 0.95rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization & Engine Instantiation
# ---------------------------------------------------------
if "analyzer" not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()
if "history" not in st.session_state:
    st.session_state.history = []
if "user_input_text" not in st.session_state:
    st.session_state.user_input_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

analyzer = st.session_state.analyzer
preprocessor = st.session_state.preprocessor

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("College NLP Mini-Project")
    
    st.markdown("---")
    st.subheader("Model Configuration")
    nlp_model = st.selectbox(
        "Select NLP Engine:",
        ["NLTK VADER", "TextBlob"],
        help="NLTK VADER is optimized for social sentiments and polarity scores. TextBlob computes polarity and subjectivity."
    )
    
    st.markdown("---")
    st.subheader("💡 NLP Model Insights")
    if nlp_model == "NLTK VADER":
        st.info(
            "**NLTK VADER** (Valence Aware Dictionary for sEntiment Reasoner):\n"
            "- Uses a curated human-validated lexicon.\n"
            "- Incorporates heuristics for capitalization, punctuation (!!!), and negations ('not good').\n"
            "- Computes Compound, Pos, Neu, Neg scores."
        )
    else:
        st.info(
            "**TextBlob Sentiment**:\n"
            "- Uses Natural Language pattern analyzer.\n"
            "- **Polarity**: Measures positivity [-1.0 to +1.0].\n"
            "- **Subjectivity**: Measures factual (0.0) vs opinion (1.0)."
        )
        
    st.markdown("---")
    st.subheader("📜 Session Controls")
    if st.button("🗑️ Clear History & Reset", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.user_input_text = ""
        st.rerun()

    st.markdown("---")
    st.caption("Developed with Python, Streamlit & NLTK • 100% Local Execution")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <span style="font-size: 2.2rem;">💬</span>
        <h1 style="margin: 0;">Sentiment Analysis App</h1>
    </div>
    <p>
        Natural Language Processing (NLP) application to analyze and classify the emotional polarity of text
        into <strong>Positive</strong>, <strong>Negative</strong>, or <strong>Neutral</strong> sentiments with detailed linguistic metrics.
    </p>
    <div>
        <span class="header-badge">✨ Rule-Based NLP</span>
        <span class="header-badge">⚡ Real-Time Classification</span>
        <span class="header-badge">🔬 Token Preprocessing Pipeline</span>
        <span class="header-badge">🔒 100% Local & Privacy-Friendly</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# App Navigation Tabs
# ---------------------------------------------------------
tab_analyze, tab_stats, tab_history, tab_guide = st.tabs([
    "📝 Live Analysis",
    "📊 Statistics & Charts",
    "📜 Session History",
    "📚 NLP Concepts & Guide"
])

# =========================================================
# TAB 1: Live Sentiment Analysis
# =========================================================
with tab_analyze:
    col_input, col_examples = st.columns([7, 5], gap="large")

    with col_examples:
        st.subheader("✨ Sample Texts")
        st.caption("Click any sample below to load and test immediately:")
        
        category = st.selectbox("Select Sample Category:", list(SAMPLE_TEXTS.keys()))
        selected_sample = st.radio(
            "Choose a sentence:",
            SAMPLE_TEXTS[category],
            label_visibility="collapsed"
        )
        
        if st.button("📥 Load Sample into Input Box", use_container_width=True):
            st.session_state.user_input_text = selected_sample
            st.rerun()

    with col_input:
        st.subheader("✍️ Enter Text for Analysis")
        user_text = st.text_area(
            "Type or paste any sentence, customer review, feedback, or tweet:",
            value=st.session_state.user_input_text,
            height=145,
            placeholder="e.g. The customer support team was incredibly helpful and solved my issue within minutes! 🚀"
        )
        
        # Word & Char counts
        word_count = len(user_text.split()) if user_text.strip() else 0
        char_count = len(user_text)
        st.caption(f"📊 Character Count: **{char_count}** | Word Count: **{word_count}**")

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.user_input_text = ""
                st.session_state.last_result = None
                st.rerun()

    # Trigger analysis
    if analyze_clicked:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text or select an example above before clicking Analyze.")
        else:
            with st.spinner("Analyzing text with NLP pipeline..."):
                result = analyzer.analyze(user_text, model_type=nlp_model)
                st.session_state.last_result = result
                st.session_state.user_input_text = user_text
                add_to_history(
                    st.session_state,
                    text=user_text,
                    sentiment=result["sentiment"],
                    score=result["polarity_score"],
                    model=result["model"]
                )

    # Display Analysis Result
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("---")
        st.subheader("🎯 Sentiment Prediction & NLP Metrics")

        card_class = f"result-card-{res['sentiment'].lower()}"
        
        st.markdown(f"""
        <div class="{card_class}">
            <div class="sentiment-title">
                <span>{res['emoji']}</span>
                <span>{res['sentiment']} Sentiment</span>
            </div>
            <p style="margin: 0; font-size: 1.05rem; opacity: 0.9;">
                Model: <strong>{res['model']}</strong> &nbsp;|&nbsp; 
                Polarity Score: <strong>{res['polarity_score']:+.4f}</strong> &nbsp;|&nbsp; 
                Confidence: <strong>{res['confidence']}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Metrics Row
        if res["model"] == "NLTK VADER":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Compound Score", f"{res['compound_score']:+.4f}", help="Overall normalized polarity (-1 to +1)")
            m2.metric("Positive Proportion", f"{res['positive_score'] * 100:.1f}%", help="Proportion of positive sentiment")
            m3.metric("Neutral Proportion", f"{res['neutral_score'] * 100:.1f}%", help="Proportion of neutral sentiment")
            m4.metric("Negative Proportion", f"{res['negative_score'] * 100:.1f}%", help="Proportion of negative sentiment")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Polarity", f"{res['polarity_score']:+.4f}", help="TextBlob polarity score (-1 to +1)")
            m2.metric("Subjectivity", f"{res['subjectivity_score']:.4f}", help="0.0 = Objective, 1.0 = Subjective")
            m3.metric("Type", res["details"]["Subjectivity Type"])

        # Visual Polarity Gauge
        gauge_score = res["polarity_score"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Polarity Gauge ({res['model']})", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': res['badge_color']},
                'steps': [
                    {'range': [-1, -0.05], 'color': "rgba(239, 68, 68, 0.25)"},
                    {'range': [-0.05, 0.05], 'color': "rgba(245, 158, 11, 0.25)"},
                    {'range': [0.05, 1], 'color': "rgba(16, 185, 129, 0.25)"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': gauge_score
                }
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Keyword Sentiment Extraction
        wb = res.get("word_breakdown", {})
        pos_words = wb.get("positive_words", [])
        neg_words = wb.get("negative_words", [])

        if pos_words or neg_words:
            st.markdown("#### 🔍 Sentiment-Bearing Lexicon Tokens Detected")
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown("**🟢 Positive Signals:**")
                if pos_words:
                    tags_html = "".join([f"<span class='word-tag-pos'>{w} ({s:+.1f})</span>" for w, s in pos_words])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.caption("No strong positive lexicon words found.")
            with kw_col2:
                st.markdown("**🔴 Negative Signals:**")
                if neg_words:
                    tags_html = "".join([f"<span class='word-tag-neg'>{w} ({s:+.1f})</span>" for w, s in neg_words])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.caption("No strong negative lexicon words found.")

        # Educational Preprocessing Pipeline Visualizer
        with st.expander("🔬 View Step-by-Step NLP Preprocessing Pipeline (Educational)", expanded=False):
            pipe = res.get("pipeline", {})
            st.markdown("This breakdown demonstrates the standard stages of an NLP pipeline before feature extraction:")
            
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"""
                <div class="pipeline-step">
                    <h4>1. Raw Input</h4>
                    <p style="color: #475569; font-size: 0.9rem;">{pipe.get('original_text', '')}</p>
                </div>
                <div class="pipeline-step">
                    <h4>2. Cleaned & Normalized Text</h4>
                    <p style="color: #475569; font-size: 0.9rem;">{pipe.get('cleaned_text', '')}</p>
                </div>
                <div class="pipeline-step">
                    <h4>3. Lowercase Conversion</h4>
                    <p style="color: #475569; font-size: 0.9rem;">{pipe.get('lowercased_text', '')}</p>
                </div>
                """, unsafe_allow_html=True)

            with p2:
                tokens = pipe.get("tokens", [])
                tokens_html = "".join([f"<span class='word-tag-token'>{t}</span>" for t in tokens[:30]])
                
                filtered = pipe.get("filtered_tokens", [])
                filtered_html = "".join([f"<span class='word-tag-token'>{t}</span>" for t in filtered[:30]])

                lemmatized = pipe.get("lemmatized_tokens", [])
                lemmatized_html = "".join([f"<span class='word-tag-token'>{t}</span>" for t in lemmatized[:30]])

                st.markdown(f"""
                <div class="pipeline-step">
                    <h4>4. Tokenization ({pipe.get('token_count', 0)} tokens)</h4>
                    <div>{tokens_html or '<em>None</em>'}</div>
                </div>
                <div class="pipeline-step">
                    <h4>5. Stopword Filtering ({pipe.get('removed_stopwords_count', 0)} removed)</h4>
                    <div>{filtered_html or '<em>None</em>'}</div>
                </div>
                <div class="pipeline-step">
                    <h4>6. Lemmatized Tokens</h4>
                    <div>{lemmatized_html or '<em>None</em>'}</div>
                </div>
                """, unsafe_allow_html=True)


# =========================================================
# TAB 2: Statistics & Distribution Charts
# =========================================================
with tab_stats:
    st.subheader("📊 Session Sentiment Analytics")
    history_df = get_history_df(st.session_state)
    stats = calculate_statistics(history_df)

    if history_df.empty:
        st.info("ℹ️ No predictions yet in this session. Analyze some sentences in the 'Live Analysis' tab to view analytics.")
    else:
        # Key Metrics Row
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Analyzed", stats["total"])
        s2.metric("Positive Texts", f"{stats['positive']} ({stats['pos_pct']}%)")
        s3.metric("Neutral Texts", f"{stats['neutral']} ({stats['neu_pct']}%)")
        s4.metric("Negative Texts", f"{stats['negative']} ({stats['neg_pct']}%)")
        s5.metric("Avg Polarity Score", f"{stats['avg_score']:+.3f}")

        st.markdown("---")
        
        # Charts Row
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 🥧 Sentiment Class Distribution")
            dist_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [stats["positive"], stats["neutral"], stats["negative"]]
            })
            
            dist_data_filtered = dist_data[dist_data["Count"] > 0]
            
            color_map = {
                "Positive": "#10B981",
                "Neutral": "#F59E0B",
                "Negative": "#EF4444"
            }

            fig_pie = px.pie(
                dist_data_filtered,
                names="Sentiment",
                values="Count",
                color="Sentiment",
                color_discrete_map=color_map,
                hole=0.45
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.markdown("#### 📊 Sentiment Counts Bar Chart")
            fig_bar = px.bar(
                dist_data,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map=color_map,
                text="Count"
            )
            fig_bar.update_layout(
                yaxis_title="Count of Predictions",
                xaxis_title="Sentiment Class",
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Progression timeline
        if len(history_df) >= 2:
            st.markdown("#### 📈 Polarity Score Progression (Session Sequence)")
            prog_df = history_df.iloc[::-1].reset_index(drop=True)
            prog_df["Index"] = range(1, len(prog_df) + 1)
            
            fig_line = px.line(
                prog_df,
                x="Index",
                y="Polarity Score",
                markers=True,
                hover_data=["Sentiment", "Text", "Time"],
                title="Sequence of Analyzed Texts"
            )
            fig_line.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Neutral Line (0.0)")
            fig_line.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_line, use_container_width=True)


# =========================================================
# TAB 3: History & Export
# =========================================================
with tab_history:
    st.subheader("📜 Session Analysis History")
    history_df = get_history_df(st.session_state)

    if history_df.empty:
        st.info("ℹ️ No texts analyzed in this session yet.")
    else:
        st.caption(f"Displaying **{len(history_df)}** records analyzed during this session:")
        
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            filter_sentiment = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Negative", "Neutral"])
        with search_col:
            search_query = st.text_input("Search within analyzed text:", placeholder="Type to filter...")

        filtered_df = history_df.copy()
        if filter_sentiment != "All":
            filtered_df = filtered_df[filtered_df["Sentiment"] == filter_sentiment]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df["Text"].str.contains(search_query, case=False, na=False)]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Text": st.column_config.TextColumn("Analyzed Text", width="large"),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Polarity Score": st.column_config.NumberColumn("Polarity Score", format="%.4f", width="small"),
                "Model": st.column_config.TextColumn("Model", width="small")
            }
        )

        export_col1, export_col2 = st.columns([2, 2])
        with export_col1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History as CSV",
                data=csv_data,
                file_name="sentiment_analysis_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with export_col2:
            if st.button("🗑️ Clear Entire History", use_container_width=True):
                st.session_state.history = []
                st.rerun()


# =========================================================
# TAB 4: NLP Educational Guide
# =========================================================
with tab_guide:
    st.subheader("📚 NLP Sentiment Analysis Concepts (Presentation Guide)")
    
    st.markdown("""
    ### 1. What is Sentiment Analysis?
    **Sentiment Analysis** (also known as *Opinion Mining*) is a subfield of Natural Language Processing (NLP) that algorithmically identifies, extracts, and quantifies emotional states and subjective information from raw natural language text.

    ---

    ### 2. How NLP Analyzes Emotional Polarity
    The text is evaluated along continuous or discrete sentiment dimensions:
    - **Positive (+1.0 to +0.05)**: Expresses satisfaction, happiness, approval, joy, or enthusiasm.
    - **Neutral (-0.05 to +0.05)**: Expresses factual statements, objective information, or lack of strong emotion.
    - **Negative (-0.05 to -1.0)**: Expresses dissatisfaction, disappointment, criticism, anger, or sadness.

    ---

    ### 3. NLP Preprocessing Stages in this Project
    | Stage | Purpose | Example Input | Example Output |
    | :--- | :--- | :--- | :--- |
    | **1. Text Cleaning** | Removes URLs, HTML tags, irregular punctuation | `<p>Check https://site.com #Great!</p>` | `Check Great` |
    | **2. Lowercasing** | Normalizes vocabulary space | `EXCELLENT service` | `excellent service` |
    | **3. Tokenization** | Splits sentences into constituent words | `"I loved it"` | `['i', 'loved', 'it']` |
    | **4. Stopword Removal** | Removes non-informative words (while retaining negations) | `['this', 'is', 'a', 'great', 'phone']` | `['great', 'phone']` |
    | **5. Lemmatization** | Reduces words to their dictionary root form | `['better', 'services', 'running']` | `['good', 'service', 'run']` |

    ---

    ### 4. Rule-Based NLP Models Used in this App
    #### A. NLTK VADER (*Valence Aware Dictionary and sEntiment Reasoner*)
    - Specially attuned to microblogging, social media, and customer reviews.
    - Uses a human-curated valence lexicon where words have scores between -4.0 and +4.0.
    - Handles:
      - **Punctuation intensity**: `"great!!!"` > `"great"`
      - **Capitalization**: `"GREAT"` > `"great"`
      - **Negation flipping**: `"not bad"` flips negative to positive.
      - **Degree adverbs**: `"very good"` amplifies polarity.

    #### B. TextBlob Sentiment Analyzer
    - Uses a pattern-matching lexicon approach.
    - Returns two scores:
      - **Polarity** in [-1.0, +1.0]
      - **Subjectivity** in [0.0, 1.0] (0.0 = objective fact, 1.0 = subjective opinion).

    ---

    ### 5. Why Local NLP without Paid APIs?
    - **Privacy**: No text is transmitted to third-party cloud servers.
    - **Zero Cost**: Runs completely offline using Python standard libraries and open-source models.
    - **Low Latency**: Instant inference on commodity laptops without API rate limits or network lag.
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.88rem; padding: 1rem 0;">
    🧠 <strong>Sentiment Analysis App</strong> • Built for NLP Presentation & Demonstration • Streamlit & NLTK
</div>
""", unsafe_allow_html=True)
