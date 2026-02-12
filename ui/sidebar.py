import streamlit as st
import torch

from core.memory import get_device, get_gpu_info, get_dynamic_limits
from core.embeddings import (
    SENTENCE_TRANSFORMER_MODELS,
    BASE_TRANSFORMER_MODELS,
    load_sentence_transformer,
    load_base_transformer,
)


def render_sidebar():
    """Render the full sidebar and return configuration values."""
    with st.sidebar:
        st.header("Configuration")

        # ---- Optional dependency check ----
        missing_deps = []
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            missing_deps.append("sentencepiece")

        if missing_deps:
            st.warning(f"Optional: `{', '.join(missing_deps)}` not installed")
            with st.expander("Click to install", expanded=False):
                st.markdown(f"""
                Phrase generation requires this package.

                **Install:**
                ```bash
                pip install {' '.join(missing_deps)} protobuf
                ```

                Then restart. Everything else works without it.
                """)

        # ---- Hardware ----
        st.subheader("Hardware")
        gpu = get_gpu_info()
        if gpu['available']:
            st.success(f"CUDA Available: {gpu['name']}")
            force_cpu = st.checkbox(
                "Force CPU (if GPU issues occur)",
                value=st.session_state.force_cpu,
                help="Check this if you encounter device mismatch errors",
            )
            if force_cpu != st.session_state.force_cpu:
                st.session_state.force_cpu = force_cpu
                st.rerun()
            st.caption(f"Current Device: {get_device(st.session_state.force_cpu)}")
        else:
            st.warning("CUDA not available - using CPU")
            st.caption("Install CUDA for faster processing")
            if not st.session_state.force_cpu:
                st.session_state.force_cpu = True

        st.markdown("---")

        # ---- File upload ----
        uploaded_file = st.file_uploader("Upload CSV with Intents", type=['csv'])

        encoding_option = st.selectbox(
            "File Encoding",
            ["Auto-detect", "UTF-8", "Windows-1252 (Excel)", "Latin-1", "UTF-16"],
            help="If file fails to load, try Windows-1252 for Excel files",
        )

        # ---- Thresholds ----
        st.subheader("Threshold Settings")

        st.markdown("**Intent-Level Threshold** (for heatmap and confusing pairs)")
        similarity_threshold = st.slider(
            "Intent similarity threshold:",
            min_value=0.50, max_value=1.0, value=0.85, step=0.05,
            help="Intent pairs above this threshold are flagged as confusing",
        )

        st.markdown("**Phrase-Level Threshold** (for individual phrase conflicts)")
        phrase_similarity_threshold = st.slider(
            "Phrase similarity threshold:",
            min_value=0.70, max_value=1.0, value=0.90, step=0.05,
            help="Individual phrases above this threshold are reported as conflicts.",
        )
        st.session_state.phrase_similarity_threshold = phrase_similarity_threshold

        with st.expander("How to Set Thresholds", expanded=False):
            st.markdown("""
            ### Threshold Guide:

            **For Large Datasets (1000+ phrases):**
            - Start with **0.95+** to see only critical issues
            - Work through those, then lower to 0.90

            **For Small Datasets (<500 phrases):**
            - **0.85** is a good starting point

            | Goal | Phrase Threshold | Why |
            |------|------------------|-----|
            | Find duplicates only | **0.98-1.00** | Shows near-identical phrases |
            | Critical issues only | **0.92-0.97** | High confusion risk |
            | Moderate cleanup | **0.85-0.91** | Balance of issues |
            | Comprehensive audit | **0.70-0.84** | Shows all potential issues |

            **Tip:** Start high (0.95), fix issues, re-run with lower threshold (0.90), repeat.
            """)

        st.caption(f"Current settings will show phrase conflicts >= {phrase_similarity_threshold:.2f}")

        # ---- Performance tuning ----
        dynamic_limits = get_dynamic_limits()

        with st.expander("Performance Tuning", expanded=False):
            st.caption("Use larger batches/chunks on machines with ample memory for faster analysis.")
            embed_batch_size = st.slider(
                "Embedding batch size", min_value=8, max_value=512, step=8,
                value=int(st.session_state.embedding_batch_size),
                help="Higher values speed up embedding but require more GPU/CPU memory.",
            )
            st.session_state.embedding_batch_size = int(embed_batch_size)

            phrase_chunk_size = st.slider(
                "Phrase similarity chunk size", min_value=64, max_value=1024, step=64,
                value=int(st.session_state.phrase_similarity_chunk_size),
                help="Larger chunks accelerate phrase comparison but increase RAM usage.",
            )
            st.session_state.phrase_similarity_chunk_size = int(phrase_chunk_size)

            if torch.cuda.is_available() and not st.session_state.force_cpu:
                mixed_precision = st.checkbox(
                    "Use mixed precision on GPU",
                    value=st.session_state.use_mixed_precision,
                    help="Enables torch.cuda.amp autocast for faster inference with lower memory.",
                )
                st.session_state.use_mixed_precision = mixed_precision
            else:
                st.session_state.use_mixed_precision = False
                st.caption("Mixed precision available when running on GPU.")

        # ---- System memory info ----
        st.divider()
        with st.expander("System Memory Info", expanded=False):
            st.metric("Available Memory", f"{dynamic_limits['available_gb']} GB")
            st.metric("Total Memory", f"{dynamic_limits['total_gb']} GB")
            st.metric("Memory Tier", dynamic_limits['memory_tier'])
            st.markdown("**Auto-Scaled Limits:**")
            st.write(f"- Max phrase conflicts: {dynamic_limits['phrase_conflict_max']:,}")
            st.write(f"- Embedding batch size: {dynamic_limits['embedding_batch_size']}")
            st.write(f"- Phrase chunk size: {dynamic_limits['phrase_chunk_size']}")
            st.info("""
            **Memory Tiers:**
            - Low (<4GB): Conservative limits for stability
            - Medium (4-8GB): Balanced performance
            - Good (8-16GB): Higher limits enabled
            - High (>16GB): Maximum performance
            """)

        # ---- Model selection ----
        st.subheader("Model Settings")

        model_approach = st.radio(
            "Model Approach:",
            ["Sentence Transformers (Fast)", "Base Transformer (Your Production Model)"],
            help="Sentence Transformers are optimized for embeddings. Base Transformer uses the exact model you're using in production.",
        )

        if model_approach == "Sentence Transformers (Fast)":
            st.session_state.model_type = 'sentence_transformer'
            st.session_state.pooling_strategy = 'mean'
            st.caption("Pre-optimized for similarity tasks")

            model_options = list(SENTENCE_TRANSFORMER_MODELS.keys())
            default_index = model_options.index("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

            model_name = st.selectbox(
                "Choose Model", model_options, index=default_index,
                help="Models ordered by use case. Top choices: MiniLM (fastest), mpnet (best quality), multilingual-mpnet (default), BGE/E5 (SOTA)",
            )

            info = SENTENCE_TRANSFORMER_MODELS[model_name]
            st.info(f"**{info['label']}** ({info['size']}): {info['description']}")

            with st.expander("Model Selection Guide", expanded=False):
                st.markdown("""
                ### Quick Recommendations:

                **For English Only:**
                - Speed priority: `all-MiniLM-L6-v2` (5x faster)
                - Quality priority: `all-mpnet-base-v2` or `BAAI/bge-base-en-v1.5`
                - Research/SOTA: `nomic-embed-text-v1.5` or `Alibaba-NLP/gte-base-en-v1.5`

                **For Multilingual:**
                - Balanced: `paraphrase-multilingual-mpnet-base-v2` (default)
                - Best quality: `BAAI/bge-m3` (slower, more memory)
                - Modern: `multilingual-e5-base` (good MIRACL scores)
                - Universal: `LaBSE` (109 languages, translation-focused)
                """)
        else:
            st.session_state.model_type = 'base_transformer'
            st.caption("Matches your production XLM-RoBERTa setup")
            model_name = st.selectbox("Choose Model", BASE_TRANSFORMER_MODELS)

            pooling_options = {
                "Mean Pooling (average all tokens)": "mean",
                "CLS Token (use first token embedding)": "cls",
            }
            pooling_choice = st.selectbox(
                "Pooling Strategy",
                list(pooling_options.keys()),
                index=0 if st.session_state.pooling_strategy == 'mean' else 1,
                help="CLS pooling can better mimic production classification setups; mean pooling is recommended for similarity.",
            )
            st.session_state.pooling_strategy = pooling_options[pooling_choice]

        # ---- Load model button ----
        if st.button("Load Model"):
            current_device = get_device(st.session_state.force_cpu)
            device_str = str(current_device)
            with st.spinner("Loading model... (this may take a moment for large models)"):
                try:
                    if st.session_state.model_type == 'sentence_transformer':
                        st.session_state.model = load_sentence_transformer(model_name, device_str)
                        st.session_state.tokenizer = None
                    else:
                        model, tokenizer = load_base_transformer(model_name, device_str)
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer

                    st.success(f"Model loaded: {model_name}")
                    st.info(f"Using: {st.session_state.model_type} on {device_str}")
                    st.session_state.analysis_results = None
                    st.session_state.embeddings = None

                    if st.session_state.model_type == 'base_transformer':
                        sample_param = next(st.session_state.model.parameters())
                        st.caption(f"Model device verified: {sample_param.device}")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.info("Try checking 'Force CPU' if you're experiencing device errors")
                    st.session_state.model = None

    return {
        'uploaded_file': uploaded_file,
        'encoding_option': encoding_option,
        'similarity_threshold': similarity_threshold,
        'phrase_similarity_threshold': phrase_similarity_threshold,
        'dynamic_limits': dynamic_limits,
        'model_name': model_name,
    }
