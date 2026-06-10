import streamlit as st
import pandas as pd

def render_info_page():
    """Render the Info / Product Overview tab"""
    
    # SECTION 1 - Hero Overview
    st.markdown("<h1 style='text-align: center;'>🛡️ Hybrid Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #888888; margin-bottom: 2rem;'>AI-Powered News Verification Platform</h4>", unsafe_allow_html=True)
    
    # Hero Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "96.46%")
        st.metric("AI Verification", "Gemini AI")
    with col2:
        st.metric("Training Articles", "59,220+")
        st.metric("Real-Time Search", "DuckDuckGo")
    with col3:
        st.metric("Models", "5-Model Ensemble")
        st.metric("API Support", "FastAPI")
        
    st.markdown("---")
    
    # SECTION 2 - How It Works
    st.subheader("⚙️ How It Works")
    
    workflow_steps = [
        "📄 Article Input",
        "🔤 Text Processing",
        "📊 TF-IDF Feature Extraction",
        "🧠 5-Model Ensemble Prediction",
        "🗣️ Linguistic Analysis",
        "🤖 Gemini AI Verification",
        "🌐 Web Cross-Reference",
        "⚖️ Final Verdict"
    ]
    
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: rgba(128,128,128,0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            {" <br><span style='font-size: 24px; color: #888888;'>⬇</span><br> ".join([f"<b>{step}</b>" for step in workflow_steps])}
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # SECTION 3 - Core Technologies
    st.subheader("🛠️ Core Technologies")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        with st.container(border=True):
            st.markdown("**Frontend**")
            st.caption("• Streamlit")
        with st.container(border=True):
            st.markdown("**Machine Learning**")
            st.caption("• Scikit-learn<br>• XGBoost<br>• LightGBM", unsafe_allow_html=True)
            
    with tech_col2:
        with st.container(border=True):
            st.markdown("**Backend**")
            st.caption("• FastAPI<br>• Uvicorn", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**NLP**")
            st.caption("• spaCy<br>• TextBlob<br>• VADER", unsafe_allow_html=True)
            
    with tech_col3:
        with st.container(border=True):
            st.markdown("**AI Layer & Web**")
            st.caption("• Google Gemini<br>• DuckDuckGo API", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Data & Vis**")
            st.caption("• SQLite<br>• Plotly & Matplotlib", unsafe_allow_html=True)

    st.markdown("---")
    
    # SECTION 4 - Features
    st.subheader("✨ Key Features")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.info("🔍 Fake News Detection")
        st.info("📦 Batch Processing")
    with f_col2:
        st.success("🤖 AI Fact Verification")
        st.success("🔌 REST API Access")
    with f_col3:
        st.warning("🌐 Real-Time News Search")
        st.warning("💾 History Tracking")
    with f_col4:
        st.error("📊 Analytics Dashboard")
        st.error("📈 Confidence Scoring")

    st.markdown("---")
    
    # SECTION 5 - Performance Metrics
    st.subheader("🎯 Performance Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Accuracy", "96.46%")
        st.progress(0.9646)
    with m_col2:
        st.metric("Precision", "95.84%")
        st.progress(0.9584)
    with m_col3:
        st.metric("Recall", "97.14%")
        st.progress(0.9714)
    with m_col4:
        st.metric("F1 Score", "96.48%")
        st.progress(0.9648)
        
    st.markdown("---")
    
    # SECTION 6 - API Endpoints
    st.subheader("🔌 API Endpoints")
    st.markdown("The platform exposes a robust FastAPI backend for programmatic access.")
    
    api_data = [
        {"Endpoint": "GET /", "Description": "API Welcome Message"},
        {"Endpoint": "GET /health", "Description": "System Health Check"},
        {"Endpoint": "POST /api/v1/analyze", "Description": "Analyze a single article"},
        {"Endpoint": "POST /api/v1/batch", "Description": "Analyze multiple articles"},
        {"Endpoint": "GET /api/v1/stats", "Description": "Retrieve usage statistics"}
    ]
    st.table(pd.DataFrame(api_data).set_index("Endpoint"))
    st.caption("🔒 **Authentication:** Requires `X-API-Key` header. | ⏱️ **Rate Limits:** 100 req/min standard.")

    st.markdown("---")
    
    # SECTION 7 - Usage Guide
    st.subheader("📖 Usage Guide")
    st.markdown("""
    <div style='background-color: rgba(128,128,128,0.1); padding: 20px; border-radius: 10px;'>
        <p><b>Step 1:</b> Paste or upload an article</p>
        <p><b>Step 2:</b> Run analysis</p>
        <p><b>Step 3:</b> Review prediction and confidence</p>
        <p><b>Step 4:</b> Verify AI and Web Search results</p>
        <p><b>Step 5:</b> Export or review history</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SECTION 8 - Footer
    st.markdown("""
    <div style='text-align: center; color: #888888; font-size: 14px;'>
        <p>Built with: <b>Python • Machine Learning • NLP • FastAPI • Streamlit • Gemini AI</b></p>
        <p>Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)
