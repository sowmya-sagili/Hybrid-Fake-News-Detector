import streamlit as st
import json
import os
import sys
import logging
import traceback
from pathlib import Path
import re
import numpy as np

# ── Logging: always visible in terminal ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("fake_news_app")

# ── joblib / sklearn safe import ─────────────────────────────────────────────
try:
    from joblib import load
    from sklearn.base import BaseEstimator, ClassifierMixin
except ImportError as _ie:
    st.error(f"❌ Missing required packages: {_ie}\n\n"
             "Run: `.venv/bin/python -m pip install scikit-learn joblib`")
    st.stop()

# Import new modules
from ui_components import render_header, render_sidebar_metrics, render_api_status
from state_management import init_session_state
from app_pages.analyze_page import render_analyze_page
from app_pages.batch_page import render_batch_page
from app_pages.dashboard_page import render_dashboard_page
from app_pages.info_page import render_info_page
from api_client import check_gnews_api, gemini_verify_claim, search_news_gnews, translate_text

# Import utils and features
try:
    from utils import clean_text, extract_features, get_conspiracy_indicators, get_sensationalism_score
except ImportError:
    st.error("❌ Critical Error: utils.py not found. Please ensure all files are present.")
    st.stop()

# ── Resolve paths relative to this file so they work regardless of CWD ──────
_HERE = Path(__file__).parent.resolve()
MODELS_DIR  = _HERE / "models"
MODEL_PATH  = MODELS_DIR / "pipeline.joblib"
CONFIG_PATH = MODELS_DIR / "config.json"

# NOTE: In production, use st.secrets or environment variables
GNEWS_API_KEY  = "84750e988d2d3e69c1c5e94293393433"
GEMINI_API_KEY = "AIzaSyBa8Txd9gDph1tMP4h7A8iNkWiNN5UrQ3Q"

# Initialize optional features
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from database import AnalysisDatabase
    from enhanced_features import (
        analyze_source, classify_topic, get_word_importance,
        get_model_predictions, highlight_suspicious_phrases,
        calculate_readability_score, extract_time_references,
        extract_domain
    )
    db = AnalysisDatabase()
    HAS_ENHANCED_FEATURES = True
except Exception as e:
    HAS_ENHANCED_FEATURES = False
    db = None

# --- Helper Functions ---

class FinalModel(BaseEstimator, ClassifierMixin):
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
    
    def predict(self, X):
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)
    
    def predict_proba(self, X):
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict_proba(X_tfidf)

@st.cache_resource
def load_model():
    """
    Attempt to load the saved model files.  All errors are logged to the
    terminal AND surfaced to the Streamlit UI instead of being silently swallowed.
    """
    model_joblib  = MODELS_DIR / "model.joblib"
    tfidf_joblib  = MODELS_DIR / "tfidf.joblib"
    opt_joblib    = MODELS_DIR / "pipeline_optimized.joblib"
    cfg_json      = MODELS_DIR / "config.json"

    log.info("load_model() called.")
    log.info(f"  models dir   : {MODELS_DIR}")
    log.info(f"  model.joblib : {model_joblib} | exists={model_joblib.exists()}")
    log.info(f"  tfidf.joblib : {tfidf_joblib} | exists={tfidf_joblib.exists()}")
    log.info(f"  config.json  : {cfg_json}     | exists={cfg_json.exists()}")

    def _load_config(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    # ── Try 1: model.joblib + tfidf.joblib (primary format) ─────────────────
    if model_joblib.exists() and tfidf_joblib.exists():
        try:
            log.info("Attempting to load model.joblib + tfidf.joblib …")
            vectorizer = load(str(tfidf_joblib))
            model      = load(str(model_joblib))
            pipe       = FinalModel(vectorizer=vectorizer, model=model)
            config     = _load_config(cfg_json)
            log.info("✅ Model loaded successfully (model.joblib + tfidf.joblib).")
            return pipe, float(config.get("threshold", 0.5)), config
        except Exception as exc:
            tb = traceback.format_exc()
            log.error(f"❌ Failed loading model.joblib / tfidf.joblib:\n{tb}")
            st.error(
                f"**Model files found but could not be loaded.**\n\n"
                f"```\n{tb}\n```\n\n"
                f"This usually means the model was trained with a different "
                f"version of scikit-learn/joblib. Run `python train_model.py` to retrain."
            )
            return None, 0.5, {}

    # ── Try 2: pipeline_optimized.joblib (legacy format) ────────────────────
    if opt_joblib.exists():
        try:
            log.info("Attempting to load pipeline_optimized.joblib …")
            model_data = load(str(opt_joblib))
            if isinstance(model_data, dict) and "vectorizer" in model_data:
                pipe = FinalModel(
                    vectorizer=model_data["vectorizer"],
                    model=model_data["model"]
                )
            else:
                pipe = model_data
            config = _load_config(cfg_json)
            log.info("✅ Model loaded successfully (pipeline_optimized.joblib).")
            return pipe, float(config.get("threshold", 0.5)), config
        except Exception as exc:
            tb = traceback.format_exc()
            log.error(f"❌ Failed loading pipeline_optimized.joblib:\n{tb}")
            st.error(f"**Legacy model failed to load.**\n\n```\n{tb}\n```")
            return None, 0.5, {}

    # ── Try 3: pipeline.joblib (original format) ─────────────────────────────
    if MODEL_PATH.exists():
        try:
            log.info("Attempting to load pipeline.joblib …")
            pipe   = load(str(MODEL_PATH))
            config = _load_config(cfg_json)
            log.info("✅ Model loaded successfully (pipeline.joblib).")
            return pipe, float(config.get("threshold", 0.5)), config
        except Exception as exc:
            tb = traceback.format_exc()
            log.error(f"❌ Failed loading pipeline.joblib:\n{tb}")
            st.error(f"**pipeline.joblib failed to load.**\n\n```\n{tb}\n```")
            return None, 0.5, {}

    # ── No model files found ─────────────────────────────────────────────────
    missing = [str(p) for p in [model_joblib, tfidf_joblib] if not p.exists()]
    msg = (
        f"**No model files found.** Missing: `{', '.join(missing)}`\n\n"
        f"**To fix, run the trainer from the project folder:**\n"
        f"```\npython train_model.py\n```\n\n"
        f"This will create `models/model.joblib` and `models/tfidf.joblib`."
    )
    log.error("❌ " + msg.replace("**", "").replace("`", "").replace("\n", " "))
    st.error(msg)
    return None, 0.5, {}

# --- Main Application ---

st.set_page_config(
    page_title='Hybrid Fake News Detector', 
    page_icon='🔍', 
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '# Hybrid Fake News Detector v2.0\nMLinformation detection with 96.46% accuracy'
    }
)

# Initialize state
init_session_state()

# Dark mode toggle
dark_mode = st.sidebar.checkbox('🌙 Dark Mode', value=False)
if dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #FAFAFA;
        }
        .stTextArea textarea {
            background-color: #262730;
            color: #FAFAFA;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stTextArea textarea {
            background-color: #F0F2F6;
            color: #000000;
        }
    </style>
    """, unsafe_allow_html=True)

# Render Header
render_header()

# Load Model
pipe, threshold, cfg = load_model()

# Sidebar
with st.sidebar:
    # Check API status
    ok, msg, details = check_gnews_api(GNEWS_API_KEY)
    render_api_status(ok, msg, HAS_GEMINI)
    
    st.markdown('---')
    st.subheader('⚙️ Settings')
    sensitivity = st.slider('Detection Sensitivity', 0.0, 1.0, threshold, 0.05)
    st.caption('Lower = more likely to flag as fake')
    
    if HAS_ENHANCED_FEATURES and db:
        st.markdown('---')
        if st.button('📜 View History', use_container_width=True):
            st.session_state['show_history'] = True

if st.session_state.get('show_history', False):
    import streamlit.components.v1 as components
    components.html("""
    <script>
    const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
    for (let i = 0; i < tabs.length; i++) {
        if (tabs[i].innerText.includes('Dashboard')) {
            tabs[i].click();
            break;
        }
    }
    setTimeout(() => {
        const headers = window.parent.document.querySelectorAll('h3');
        for (let i = 0; i < headers.length; i++) {
            if (headers[i].innerText.includes('Recent Analyses')) {
                headers[i].scrollIntoView({behavior: 'smooth', block: 'start'});
                break;
            }
        }
    }, 200);
    </script>
    """, height=0)
    st.session_state['show_history'] = False

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(['🔍 Analyze', '📦 Batch Upload', 'ℹ️ Info', '📊 Dashboard'])


with tab1:
    # Pass API keys implicitly via bound functions or explicit args
    # We'll wrap the API calls to include keys
    verify_func = lambda text: gemini_verify_claim(text, GEMINI_API_KEY)
    search_func = lambda query, max_results=10: search_news_gnews(query, GNEWS_API_KEY, max_results)
    translate_func = lambda text: translate_text(text, 'en', GEMINI_API_KEY)
    
    render_analyze_page(pipe, sensitivity, db, HAS_ENHANCED_FEATURES, verify_func, search_func, translate_func)

with tab2:
    render_batch_page(pipe, db)

with tab3:
    render_info_page()

with tab4:
    render_dashboard_page(db)
