"""
Intent Similarity Analyzer with Semantic Search
================================================
All processing runs locally -- no data leaves your machine.

Run:
    streamlit run app.py
"""
import logging
import streamlit as st
import pandas as pd

from core.memory import get_dynamic_limits
from utils.data_loader import load_intent_dataframe, build_intents_dict
from utils.persistence import save_project, load_project, list_projects, delete_project
from ui.sidebar import render_sidebar
from ui.tab_similarity import render_tab_similarity
from ui.tab_search import render_tab_search

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="intent_analyzer.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Intent Similarity Analyzer with Semantic Search", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_defaults = {
    'data': None,
    'embeddings': None,
    'model': None,
    'tokenizer': None,
    'model_type': None,
    'force_cpu': False,
    'baseline_results': None,
    'analysis_results': None,
    'embedding_batch_size': 32,
    'phrase_similarity_chunk_size': 256,
    'use_mixed_precision': True,
    'uploaded_file_signature': None,
    'last_successful_encoding': None,
    'data_load_message': None,
    'last_selected_encoding_option': None,
    'generated_phrases': {},
    'pooling_strategy': 'mean',
    'search_history': [],
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
sidebar_cfg = render_sidebar()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Intent Similarity Analyzer with Semantic Search")
st.markdown("Identify confusing intents, audit your dataset, and find where to add new phrases")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Similarity Analysis", "Semantic Search & Audit", "Project Management"])

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
uploaded_file = sidebar_cfg['uploaded_file']
encoding_option = sidebar_cfg['encoding_option']

if uploaded_file is not None:
    df = st.session_state.data
    file_signature = (uploaded_file.name, uploaded_file.size, getattr(uploaded_file, "type", None))
    needs_reload = (
        st.session_state.uploaded_file_signature != file_signature
        or st.session_state.last_selected_encoding_option != encoding_option
        or df is None
    )

    if needs_reload:
        try:
            df, detected_encoding = load_intent_dataframe(uploaded_file, encoding_option)
            st.session_state.data = df
            st.session_state.analysis_results = None
            st.session_state.embeddings = None
            st.session_state.uploaded_file_signature = file_signature
            st.session_state.last_selected_encoding_option = encoding_option
            st.session_state.last_successful_encoding = detected_encoding
            if encoding_option == "Auto-detect" and detected_encoding:
                message = f"Loaded {len(df.columns)} intents (detected encoding: {detected_encoding})"
            else:
                display_encoding = detected_encoding or encoding_option
                message = f"Loaded {len(df.columns)} intents (encoding: {display_encoding})"
            st.session_state.data_load_message = message
            logger.info(message)
        except Exception as e:
            st.session_state.data = None
            st.session_state.uploaded_file_signature = None
            st.session_state.data_load_message = None
            st.error(f"Error loading file: {str(e)}")
            st.info("Try selecting a different encoding from the sidebar (usually Windows-1252 for Excel files)")
            logger.exception("Failed to load CSV")
            st.stop()
    else:
        df = st.session_state.data

    if st.session_state.data is None:
        st.stop()

    if st.session_state.data_load_message:
        st.success(st.session_state.data_load_message)

    intents = build_intents_dict(df)
    st.info(f"Total phrases across all intents: {sum(len(p) for p in intents.values())}")

    with tab1:
        render_tab_similarity(df, intents, sidebar_cfg)

    with tab2:
        render_tab_search(intents, sidebar_cfg)

else:
    with tab1:
        st.info("Upload a CSV file to begin")
        st.markdown("""
        ### Expected CSV Format:

        | greeting | farewell | book_flight |
        |----------|----------|-------------|
        | hello | goodbye | I want to book a flight |
        | hi there | see you later | book me a ticket |
        | good morning | bye | reserve a flight |

        **Important:**
        - Each **column** represents an intent
        - **Rows** contain keyphrases for that intent
        - Empty cells are okay

        ### Features:

        #### Similarity Analysis Tab
        - Intent-to-intent similarity matrix with heatmap
        - **Intent Health Dashboard** with per-intent scores
        - **t-SNE phrase embedding scatter plot** to visualise clusters
        - Phrase-level conflict detection with **FAISS-accelerated** search
        - **TF-IDF keyword analysis** with bigrams
        - Actionable recommendations (merge / differentiate / rephrase)
        - Excel/CSV export of results

        #### Semantic Search & Audit Tab
        - Single-phrase semantic search with similarity scores
        - **Batch search** -- audit dozens of phrases at once
        - Get actionable recommendations based on similarity patterns
        - Export detailed reports
        - Track search history

        #### Project Management Tab
        - **Save & load** analysis sessions locally
        - Resume work across restarts
        """)

    intents = {}

# ---------------------------------------------------------------------------
# Tab 3: Project management (new)
# ---------------------------------------------------------------------------
with tab3:
    st.header("Project Management")
    st.info("Save and load analysis sessions locally. All data stays on your machine.")

    col_save, col_load = st.columns(2)

    with col_save:
        st.subheader("Save Current Session")
        project_name = st.text_input("Project name:", value="my_analysis", key="save_project_name")
        if st.button("Save Project", key="save_project_btn"):
            if st.session_state.data is not None:
                try:
                    path = save_project(
                        project_name,
                        st.session_state.data,
                        st.session_state.analysis_results,
                        sidebar_cfg.get('model_name', 'unknown'),
                        st.session_state.model_type,
                        baseline_results=st.session_state.baseline_results,
                        search_history=st.session_state.search_history,
                        generated_phrases=st.session_state.generated_phrases,
                    )
                    st.success(f"Project saved to: {path}")
                    logger.info("Project saved: %s", path)
                except Exception as e:
                    st.error(f"Error saving project: {str(e)}")
                    logger.exception("Failed to save project")
            else:
                st.warning("No data to save. Upload a CSV first.")

    with col_load:
        st.subheader("Load Saved Project")
        projects = list_projects()
        if projects:
            proj_df = pd.DataFrame(projects)
            st.dataframe(proj_df, width='stretch')

            selected_project = st.selectbox("Select project:", [p['name'] for p in projects], key="load_project_select")

            col_load_btn, col_del_btn = st.columns(2)
            with col_load_btn:
                if st.button("Load Project", key="load_project_btn"):
                    try:
                        project = load_project(selected_project)
                        st.session_state.data = project['data']
                        st.session_state.analysis_results = project.get('analysis_results')
                        st.session_state.baseline_results = project.get('baseline_results')
                        st.session_state.search_history = project.get('search_history', [])
                        st.session_state.generated_phrases = project.get('generated_phrases', {})
                        st.session_state.data_load_message = f"Loaded project: {selected_project} (saved {project.get('saved_at', 'unknown')})"

                        # Restore embeddings from analysis results
                        if st.session_state.analysis_results and 'embeddings' in st.session_state.analysis_results:
                            import numpy as np
                            emb = st.session_state.analysis_results['embeddings']
                            if isinstance(emb, list):
                                emb = np.array(emb)
                            st.session_state.embeddings = emb

                        st.success(f"Project '{selected_project}' loaded! Note: you'll need to reload the model ({project.get('model_name', 'unknown')}) for new analysis.")
                        logger.info("Project loaded: %s", selected_project)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading project: {str(e)}")
                        logger.exception("Failed to load project")

            with col_del_btn:
                if st.button("Delete Project", key="delete_project_btn"):
                    delete_project(selected_project)
                    st.success(f"Deleted project: {selected_project}")
                    st.rerun()
        else:
            st.info("No saved projects yet. Save your current session to get started.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("Built for Intent Classification | Powered by Sentence Transformers | All data stays local")
st.caption("Excel export requires openpyxl (`pip install openpyxl`). FAISS acceleration requires faiss-cpu (`pip install faiss-cpu`).")
