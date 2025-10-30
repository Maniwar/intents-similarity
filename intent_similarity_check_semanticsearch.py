import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from collections import Counter
import re
from io import BytesIO
from contextlib import nullcontext
import json
import psutil

# Page config
st.set_page_config(page_title="Intent Similarity Analyzer with Semantic Search", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'force_cpu' not in st.session_state:
    st.session_state.force_cpu = False
if 'baseline_results' not in st.session_state:
    st.session_state.baseline_results = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'embedding_batch_size' not in st.session_state:
    st.session_state.embedding_batch_size = 32
if 'phrase_similarity_chunk_size' not in st.session_state:
    st.session_state.phrase_similarity_chunk_size = 256
if 'use_mixed_precision' not in st.session_state:
    st.session_state.use_mixed_precision = True
if 'uploaded_file_signature' not in st.session_state:
    st.session_state.uploaded_file_signature = None
if 'last_successful_encoding' not in st.session_state:
    st.session_state.last_successful_encoding = None
if 'data_load_message' not in st.session_state:
    st.session_state.data_load_message = None
if 'last_selected_encoding_option' not in st.session_state:
    st.session_state.last_selected_encoding_option = None
if 'generated_phrases' not in st.session_state:
    st.session_state.generated_phrases = {}
if 'pooling_strategy' not in st.session_state:
    st.session_state.pooling_strategy = 'mean'
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Check CUDA availability and set device
def get_device():
    if st.session_state.force_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Dynamic memory-based limits
def get_dynamic_limits():
    """Calculate dynamic processing limits based on available system memory"""
    try:
        # Get available memory in GB
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        total_gb = memory.total / (1024 ** 3)

        # Calculate phrase conflict limit based on available memory
        # Rough estimation: each conflict record ~500 bytes
        # Leave 2GB for system/other processes, use 50% of remaining available memory
        usable_gb = max(0.5, (available_gb - 2) * 0.5)

        # Conservative limits based on memory tiers
        if available_gb < 4:
            # Low memory system (<4GB available)
            phrase_conflict_max = 5_000
            embedding_batch_size = 16
            phrase_chunk_size = 128
        elif available_gb < 8:
            # Medium memory system (4-8GB available)
            phrase_conflict_max = 25_000
            embedding_batch_size = 32
            phrase_chunk_size = 256
        elif available_gb < 16:
            # Good memory system (8-16GB available)
            phrase_conflict_max = 100_000
            embedding_batch_size = 64
            phrase_chunk_size = 512
        else:
            # High memory system (>16GB available)
            phrase_conflict_max = 500_000
            embedding_batch_size = 128
            phrase_chunk_size = 1024

        return {
            'phrase_conflict_max': phrase_conflict_max,
            'embedding_batch_size': embedding_batch_size,
            'phrase_chunk_size': phrase_chunk_size,
            'available_gb': round(available_gb, 2),
            'total_gb': round(total_gb, 2),
            'memory_tier': 'Low' if available_gb < 4 else 'Medium' if available_gb < 8 else 'Good' if available_gb < 16 else 'High'
        }
    except Exception as e:
        # Fallback to conservative defaults if psutil fails
        return {
            'phrase_conflict_max': 10_000,
            'embedding_batch_size': 32,
            'phrase_chunk_size': 256,
            'available_gb': 0,
            'total_gb': 0,
            'memory_tier': 'Unknown'
        }

# Get dynamic limits on startup
DYNAMIC_LIMITS = get_dynamic_limits()
PHRASE_CONFLICT_MAX = DYNAMIC_LIMITS['phrase_conflict_max']

# Helper function for extracting keywords
def extract_keywords(text):
    """Extract meaningful keywords from text"""
    # Simple keyword extraction - remove common words
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                  'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                  'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                  'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                  'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
                  'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                  'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                  'under', 'again', 'further', 'then', 'once', 'can', 'want', 'need', 'please'}

    words = re.findall(r'\b[a-z]+\b', text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords

def analyze_keyword_overlap(intent1_phrases, intent2_phrases):
    """Analyze keyword overlap between two intents"""
    keywords1 = []
    for phrase in intent1_phrases:
        keywords1.extend(extract_keywords(phrase))

    keywords2 = []
    for phrase in intent2_phrases:
        keywords2.extend(extract_keywords(phrase))

    counter1 = Counter(keywords1)
    counter2 = Counter(keywords2)

    # Find common keywords
    common_keywords = set(counter1.keys()) & set(counter2.keys())

    overlap_score = len(common_keywords) / max(len(set(keywords1)), len(set(keywords2)), 1)

    return {
        'common_keywords': sorted(common_keywords),
        'overlap_score': overlap_score,
        'intent1_unique': sorted(set(counter1.keys()) - common_keywords)[:10],
        'intent2_unique': sorted(set(counter2.keys()) - common_keywords)[:10]
    }

# Helper function for mean pooling
def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Unified encoding function
def encode_texts(
    texts,
    model,
    tokenizer=None,
    model_type='sentence_transformer',
    device_obj=None,
    batch_size=32,
    normalize_embeddings=True,
    use_mixed_precision=True,
    show_progress=False,
    pooling_strategy='mean'
):
    """Encode texts using either sentence-transformers or base transformer models"""
    if device_obj is None:
        device_obj = torch.device('cpu')
    batch_size = max(int(batch_size), 1)

    if model_type == 'sentence_transformer':
        # Sentence transformers handle device internally
        # Convert device object to string for sentence-transformers
        device_str = str(device_obj)
        if device_obj.type != 'cpu':
            model = model.to(device_obj)
        encode_kwargs = {
            "device": device_str,
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": show_progress and len(texts) > batch_size
        }
        # Only pass normalize flag if available (newer versions support it)
        if normalize_embeddings:
            encode_kwargs["normalize_embeddings"] = True
        try:
            return model.encode(texts, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("normalize_embeddings", None)
            return model.encode(texts, **encode_kwargs)
    else:  # base transformer
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        if model_device != device_obj:
            model.to(device_obj)

        total_texts = len(texts)
        embeddings_list = []
        autocast_enabled = use_mixed_precision and device_obj.type == 'cuda'
        autocast_ctx = torch.cuda.amp.autocast if autocast_enabled else nullcontext

        for start in range(0, total_texts, batch_size):
            end = min(start + batch_size, total_texts)
            batch_texts = texts[start:end]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )

            # Move input tensors to the same device as model
            encoded_input = {k: v.to(device_obj) for k, v in encoded_input.items()}

            with torch.no_grad():
                with autocast_ctx():
                    model_output = model(**encoded_input)

            token_embeddings = model_output.last_hidden_state if hasattr(model_output, "last_hidden_state") else model_output[0]

            if pooling_strategy == 'cls':
                batch_embeddings = token_embeddings[:, 0, :]
            elif pooling_strategy == 'mean':
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

            if normalize_embeddings:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings_list.append(batch_embeddings.detach().cpu())

        if not embeddings_list:
            return np.array([])

        return torch.cat(embeddings_list, dim=0).numpy()

ENCODING_MAP = {
    "Auto-detect": None,
    "UTF-8": "utf-8",
    "Windows-1252 (Excel)": "cp1252",
    "Latin-1": "latin-1",
    "UTF-16": "utf-16"
}

AUTO_DETECT_ENCODINGS = ['utf-8', 'cp1252', 'latin-1', 'utf-16']


def load_intent_dataframe(uploaded_file, encoding_option):
    """Load uploaded CSV into dataframe using selected or detected encoding."""
    if uploaded_file is None:
        return None, None

    selected_encoding = ENCODING_MAP.get(encoding_option)

    if selected_encoding is None:
        df = None
        successful_encoding = None
        for enc in AUTO_DETECT_ENCODINGS:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                successful_encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        uploaded_file.seek(0)
        if df is None:
            raise UnicodeError("Could not detect file encoding.")
        return df, successful_encoding

    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, encoding=selected_encoding)
    uploaded_file.seek(0)
    return df, selected_encoding

PHRASE_SIMILARITY_MIN = 0.70
PHRASE_CONFLICT_MAX = 10000

def perform_analysis(
    intents,
    model,
    tokenizer,
    model_type,
    device_obj,
    embed_batch_size=32,
    phrase_chunk_size=256,
    use_mixed_precision=True,
    show_progress=True,
    pooling_strategy='mean'
):
    """Run full embedding and similarity analysis once and cache results."""
    all_phrases = []
    phrase_to_intent = []

    for intent_name, phrases in intents.items():
        all_phrases.extend(phrases)
        phrase_to_intent.extend([intent_name] * len(phrases))

    embeddings = encode_texts(
        all_phrases,
        model,
        tokenizer,
        model_type,
        device_obj=device_obj,
        batch_size=embed_batch_size,
        normalize_embeddings=True,
        use_mixed_precision=use_mixed_precision,
        show_progress=show_progress,
        pooling_strategy=pooling_strategy
    )

    # Map intent -> indices for averaging
    intent_indices = {}
    for idx, intent_name in enumerate(phrase_to_intent):
        intent_indices.setdefault(intent_name, []).append(idx)

    intent_names = list(intent_indices.keys())
    intent_embeddings = {
        intent_name: np.mean(embeddings[idx_list], axis=0)
        for intent_name, idx_list in intent_indices.items()
    }
    intent_emb_matrix = np.array([intent_embeddings[name] for name in intent_names])
    intent_sim_matrix = cosine_similarity(intent_emb_matrix)

    # Phrase-level similarities (cross-intent) at base threshold
    phrase_conflicts = []
    total_phrases = len(all_phrases)
    phrase_chunk_size = max(int(phrase_chunk_size), 1)
    limit_reached = False
    processed = 0
    progress_bar = st.progress(0) if show_progress and total_phrases > 0 else None

    for start_idx in range(0, total_phrases, phrase_chunk_size):
        end_idx = min(start_idx + phrase_chunk_size, total_phrases)
        chunk_embeddings = embeddings[start_idx:end_idx]
        similarity_block = cosine_similarity(chunk_embeddings, embeddings)

        for row_offset, phrase_index in enumerate(range(start_idx, end_idx)):
            phrase_intent = phrase_to_intent[phrase_index]
            row_similarities = similarity_block[row_offset]

            for other_index in range(phrase_index + 1, total_phrases):
                other_intent = phrase_to_intent[other_index]
                if other_intent == phrase_intent:
                    continue

                sim = row_similarities[other_index]
                if sim >= PHRASE_SIMILARITY_MIN:
                    note = '⚠️ EXACT DUPLICATE' if all_phrases[phrase_index] == all_phrases[other_index] else ''
                    phrase_conflicts.append({
                        'Phrase': all_phrases[phrase_index],
                        'Intent': phrase_intent,
                        'Similar To': all_phrases[other_index],
                        'Other Intent': other_intent,
                        'Similarity': sim,
                        'Note': note
                    })

                    if len(phrase_conflicts) >= PHRASE_CONFLICT_MAX:
                        limit_reached = True
                        break

            if limit_reached:
                break

        processed += (end_idx - start_idx)
        if progress_bar:
            progress_bar.progress(min(processed / total_phrases, 0.99))

        if limit_reached:
            break

    if progress_bar:
        progress_bar.progress(1.0)
        progress_bar.empty()

    results = {
        'all_phrases': all_phrases,
        'phrase_to_intent': phrase_to_intent,
        'embeddings': embeddings,
        'intent_names': intent_names,
        'intent_sim_matrix': intent_sim_matrix,
        'phrase_conflicts': phrase_conflicts,
        'phrase_conflict_limit_reached': limit_reached,
        'base_phrase_threshold': PHRASE_SIMILARITY_MIN,
        'embed_batch_size': embed_batch_size,
        'phrase_chunk_size': phrase_chunk_size,
        'pooling_strategy': pooling_strategy
    }

    return results

# Function for semantic search
def semantic_search(query, embeddings, all_phrases, phrase_to_intent, model, tokenizer, model_type, device_obj, pooling_strategy, threshold=0.7, top_k=100):
    """Perform semantic search on all phrases"""
    # Encode the search query
    query_embedding = encode_texts(
        [query],
        model,
        tokenizer,
        model_type,
        device_obj=device_obj,
        batch_size=1,
        normalize_embeddings=True,
        use_mixed_precision=device_obj.type == 'cuda',
        pooling_strategy=pooling_strategy
    )

    # Calculate similarities with all phrases
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Create results
    search_results = []
    for idx, (phrase, intent, similarity) in enumerate(zip(all_phrases, phrase_to_intent, similarities)):
        if similarity >= threshold:
            search_results.append({
                'similarity': similarity,
                'phrase': phrase,
                'intent': intent,
                'idx': idx
            })

    # Sort by similarity
    search_results.sort(key=lambda x: x['similarity'], reverse=True)

    # Limit to top_k
    search_results = search_results[:top_k]

    return search_results

# Title
st.title("🎯 Intent Similarity Analyzer with Semantic Search")
st.markdown("Identify confusing intents, audit your dataset, and find where to add new phrases")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📊 Similarity Analysis", "🔍 Semantic Search & Audit"])

# Sidebar (common for both tabs)
with st.sidebar:
    st.header("⚙️ Configuration")

    # Check for optional dependencies
    missing_deps = []
    try:
        import sentencepiece
    except ImportError:
        missing_deps.append("sentencepiece")

    if missing_deps:
        st.warning(f"⚠️ Optional: `{', '.join(missing_deps)}` not installed")
        with st.expander("ℹ️ Click to install", expanded=False):
            st.markdown(f"""
            Phrase generation requires this package.

            **Install:**
            ```bash
            pip install {' '.join(missing_deps)} protobuf
            ```

            Then restart. Everything else works without it.
            """)

    # GPU Status
    st.subheader("🖥️ Hardware")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"✅ CUDA Available: {gpu_name}")

        # Force CPU option
        force_cpu = st.checkbox(
            "Force CPU (if GPU issues occur)",
            value=st.session_state.force_cpu,
            help="Check this if you encounter device mismatch errors"
        )
        if force_cpu != st.session_state.force_cpu:
            st.session_state.force_cpu = force_cpu
            st.rerun()

        current_device = get_device()
        st.caption(f"Current Device: {current_device}")
    else:
        st.warning("⚠️ CUDA not available - using CPU")
        st.caption("Install CUDA for faster processing")
        if not st.session_state.force_cpu:
            st.session_state.force_cpu = True

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV with Intents", type=['csv'])

    # Encoding selector
    encoding_option = st.selectbox(
        "File Encoding",
        ["Auto-detect", "UTF-8", "Windows-1252 (Excel)", "Latin-1", "UTF-16"],
        help="If file fails to load, try Windows-1252 for Excel files"
    )

    # Similarity threshold
    st.subheader("⚙️ Threshold Settings")

    st.markdown("**Intent-Level Threshold** (for heatmap and confusing pairs)")
    similarity_threshold = st.slider(
        "Intent similarity threshold:",
        min_value=0.50,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Intent pairs above this threshold are flagged as confusing"
    )

    st.markdown("**Phrase-Level Threshold** (for individual phrase conflicts)")
    phrase_similarity_threshold = st.slider(
        "Phrase similarity threshold:",
        min_value=0.70,
        max_value=1.0,
        value=0.90,
        step=0.05,
        help="Individual phrases above this threshold are reported as conflicts. Higher = fewer conflicts shown."
    )

    # Store in session state
    if 'phrase_similarity_threshold' not in st.session_state:
        st.session_state.phrase_similarity_threshold = phrase_similarity_threshold
    else:
        st.session_state.phrase_similarity_threshold = phrase_similarity_threshold

    # Guidance
    with st.expander("💡 How to Set Thresholds", expanded=False):
        st.markdown("""
        ### Threshold Guide:

        **For Large Datasets (1000+ phrases):**
        - Start with **0.95+** to see only critical issues
        - Work through those, then lower to 0.90
        - Gradually decrease as you fix issues

        **For Small Datasets (<500 phrases):**
        - **0.85** is a good starting point
        - Shows moderate to high similarity

        **Recommended Settings by Goal:**

        | Goal | Phrase Threshold | Why |
        |------|------------------|-----|
        | Find duplicates only | **0.98-1.00** | Shows near-identical phrases |
        | Critical issues only | **0.92-0.97** | High confusion risk |
        | Moderate cleanup | **0.85-0.91** | Balance of issues |
        | Comprehensive audit | **0.70-0.84** | Shows all potential issues |

        **💡 Pro Tip:** Start high (0.95), fix issues, re-run with lower threshold (0.90), repeat.
        """)

    st.caption(f"💡 Current settings will show phrase conflicts ≥ {phrase_similarity_threshold:.2f}")

    with st.expander("⚡ Performance Tuning", expanded=False):
        st.caption("Use larger batches/chunks on machines with ample memory for faster analysis.")
        embed_batch_size = st.slider(
            "Embedding batch size",
            min_value=8,
            max_value=512,
            step=8,
            value=int(st.session_state.embedding_batch_size),
            help="Higher values speed up embedding but require more GPU/CPU memory."
        )
        st.session_state.embedding_batch_size = int(embed_batch_size)

        phrase_chunk_size = st.slider(
            "Phrase similarity chunk size",
            min_value=64,
            max_value=1024,
            step=64,
            value=int(st.session_state.phrase_similarity_chunk_size),
            help="Larger chunks accelerate phrase comparison but increase RAM usage."
        )
        st.session_state.phrase_similarity_chunk_size = int(phrase_chunk_size)

        if torch.cuda.is_available() and not st.session_state.force_cpu:
            mixed_precision = st.checkbox(
                "Use mixed precision on GPU",
                value=st.session_state.use_mixed_precision,
                help="Enables torch.cuda.amp autocast for faster inference with lower memory."
            )
            st.session_state.use_mixed_precision = mixed_precision
        else:
            st.session_state.use_mixed_precision = False
            st.caption("Mixed precision available when running on GPU.")

    # System memory information
    st.divider()
    with st.expander("💾 System Memory Info", expanded=False):
        mem_info = DYNAMIC_LIMITS
        st.metric("Available Memory", f"{mem_info['available_gb']} GB")
        st.metric("Total Memory", f"{mem_info['total_gb']} GB")
        st.metric("Memory Tier", mem_info['memory_tier'])

        st.markdown("**Auto-Scaled Limits:**")
        st.write(f"- Max phrase conflicts: {mem_info['phrase_conflict_max']:,}")
        st.write(f"- Embedding batch size: {mem_info['embedding_batch_size']}")
        st.write(f"- Phrase chunk size: {mem_info['phrase_chunk_size']}")

        st.info("""
        **Memory Tiers:**
        - 🔴 Low (<4GB): Conservative limits for stability
        - 🟡 Medium (4-8GB): Balanced performance
        - 🟢 Good (8-16GB): Higher limits enabled
        - 🔵 High (>16GB): Maximum performance

        These limits auto-scale based on your available RAM to prevent crashes while maximizing analysis depth.
        """)

    # Model selection
    st.subheader("Model Settings")

    model_approach = st.radio(
        "Model Approach:",
        ["Sentence Transformers (Fast)", "Base Transformer (Your Production Model)"],
        help="Sentence Transformers are optimized for embeddings. Base Transformer uses the exact model you're using in production."
    )

    if model_approach == "Sentence Transformers (Fast)":
        st.session_state.model_type = 'sentence_transformer'
        st.session_state.pooling_strategy = 'mean'
        st.caption("✅ Pre-optimized for similarity tasks")

        # Model descriptions for help text
        model_descriptions = {
            "sentence-transformers/all-MiniLM-L6-v2": "⚡ Ultra-fast & lightweight (80MB) | Best for: Quick prototyping, real-time apps, limited resources | Speed: 14k sentences/sec | Quality: 85% STS-B | English only",
            "sentence-transformers/all-mpnet-base-v2": "🏆 Best quality-speed balance (420MB) | Best for: Production use, high accuracy needed | Speed: 4k sentences/sec | Quality: 88% STS-B | English only",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "🌍 Multilingual champion (1.1GB) | Best for: 50+ languages, international datasets | Based on mpnet | Quality: Excellent cross-lingual | Current default",
            "BAAI/bge-base-en-v1.5": "🚀 SOTA English (440MB) | Best for: Latest research, high accuracy | Top MTEB scores | Instruction-aware embeddings | English only",
            "BAAI/bge-m3": "🌐 Multi-everything (2.3GB) | Best for: Multi-lingual + multi-granularity | 100+ languages | Self-distilled | Slower but powerful",
            "intfloat/e5-base-v2": "⚖️ Balanced performer (440MB) | Best for: General purpose, good speed-quality tradeoff | Strong on retrieval tasks | English only",
            "intfloat/multilingual-e5-base": "🌏 E5 multilingual (1.1GB) | Best for: Multilingual with E5 architecture | 100+ languages | Good MIRACL scores",
            "nomic-ai/nomic-embed-text-v1.5": "🔬 Research leader (550MB) | Best for: Cutting-edge performance, diverse inputs | Top BEIR scores | Multimodal-ready",
            "Alibaba-NLP/gte-base-en-v1.5": "🎯 Precision model (440MB) | Best for: High-quality English embeddings | Angle-optimized | Strong retrieval",
            "sentence-transformers/xlm-r-large": "🔧 Legacy large (2.2GB) | Best for: Compatibility, established workflows | XLM-RoBERTa base | 100 languages",
            "sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens": "📊 NLI-trained (1.1GB) | Best for: Natural language inference tasks | Trained on STS benchmarks",
            "sentence-transformers/LaBSE": "🔄 Universal encoder (1.8GB) | Best for: 109 languages, translation pairs | Parallel corpus trained | Cross-lingual search"
        }

        model_options = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-m3",
            "intfloat/e5-base-v2",
            "intfloat/multilingual-e5-base",
            "nomic-ai/nomic-embed-text-v1.5",
            "Alibaba-NLP/gte-base-en-v1.5",
            "sentence-transformers/xlm-r-large",
            "sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens",
            "sentence-transformers/LaBSE"
        ]

        # Find default index (multilingual mpnet)
        default_index = model_options.index("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

        model_name = st.selectbox(
            "Choose Model",
            model_options,
            index=default_index,
            help="Models ordered by use case. Top choices: MiniLM (fastest), mpnet (best quality), multilingual-mpnet (current default), BGE/E5 (SOTA 2025)"
        )

        # Show detailed info about selected model
        if model_name in model_descriptions:
            st.info(f"**{model_name.split('/')[-1]}**: {model_descriptions[model_name]}")

        # Additional guidance
        with st.expander("📘 Model Selection Guide", expanded=False):
            st.markdown("""
            ### Quick Recommendations:

            **For English Only:**
            - 🏃 Speed priority: `all-MiniLM-L6-v2` (5x faster)
            - 🎯 Quality priority: `all-mpnet-base-v2` or `BAAI/bge-base-en-v1.5`
            - 🔬 Research/SOTA: `nomic-embed-text-v1.5` or `Alibaba-NLP/gte-base-en-v1.5`

            **For Multilingual:**
            - ⚖️ Balanced: `paraphrase-multilingual-mpnet-base-v2` (current default)
            - 🌐 Best quality: `BAAI/bge-m3` (slower, more memory)
            - 🚀 Modern: `multilingual-e5-base` (good MIRACL scores)
            - 🔄 Universal: `LaBSE` (109 languages, translation-focused)

            ### Performance Notes:
            - **Size**: Larger models = better quality but slower & more memory
            - **Speed**: MiniLM is ~5x faster than mpnet variants
            - **MTEB Scores**: BGE, E5, Nomic, and GTE lead 2025 benchmarks
            - **Languages**: Check if your data needs multilingual support

            ### Latest 2025 Models:
            The BGE, E5, Nomic, and GTE series represent state-of-the-art as of 2025,
            trained on massive datasets and optimized for retrieval tasks.
            """)
    else:
        st.session_state.model_type = 'base_transformer'
        st.caption("✅ Matches your production XLM-RoBERTa setup")
        model_name = st.selectbox(
            "Choose Model",
            [
                "xlm-roberta-large",
                "xlm-roberta-base",
                "FacebookAI/xlm-roberta-large",
                "FacebookAI/xlm-roberta-base"
            ]
        )
        pooling_options = {
            "Mean Pooling (average all tokens)": "mean",
            "CLS Token (use first token embedding)": "cls"
        }
        pooling_choice = st.selectbox(
            "Pooling Strategy",
            list(pooling_options.keys()),
            index=0 if st.session_state.pooling_strategy == 'mean' else 1,
            help="CLS pooling can better mimic production classification setups; mean pooling is recommended for similarity."
        )
        st.session_state.pooling_strategy = pooling_options[pooling_choice]

    if st.button("Load Model"):
        current_device = get_device()
        with st.spinner("Loading model... (this may take a moment for large models)"):
            try:
                if st.session_state.model_type == 'sentence_transformer':
                    st.session_state.model = SentenceTransformer(model_name)
                    # Move sentence transformer to GPU if available
                    st.session_state.model = st.session_state.model.to(current_device)
                    st.session_state.tokenizer = None
                else:
                    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    # Load model and move to target device
                    st.session_state.model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    )
                    # Explicitly move all components to target device
                    st.session_state.model = st.session_state.model.to(current_device)
                    st.session_state.model.eval()

                    # Verify all parameters are on correct device
                    for name, param in st.session_state.model.named_parameters():
                        if param.device != current_device:
                            param.data = param.data.to(current_device)
                            if param._grad is not None:
                                param._grad.data = param._grad.data.to(current_device)

                device_str = str(current_device)
                st.success(f"✅ Model loaded: {model_name}")
                st.info(f"Using: {st.session_state.model_type} on {device_str}")
                st.session_state.analysis_results = None
                st.session_state.embeddings = None

                # Verify device placement
                if st.session_state.model_type == 'base_transformer':
                    sample_param = next(st.session_state.model.parameters())
                    st.caption(f"✓ Model device verified: {sample_param.device}")

            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("💡 Try checking 'Force CPU' if you're experiencing device errors")
                st.session_state.model = None

# Main content - Process data if uploaded
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
                message = f"✅ Loaded {len(df.columns)} intents (detected encoding: {detected_encoding})"
            else:
                display_encoding = detected_encoding or encoding_option
                message = f"✅ Loaded {len(df.columns)} intents (encoding: {display_encoding})"
            st.session_state.data_load_message = message
        except Exception as e:
            st.session_state.data = None
            st.session_state.uploaded_file_signature = None
            st.session_state.data_load_message = None
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Try selecting a different encoding from the sidebar (usually Windows-1252 for Excel files)")
            st.stop()
    else:
        df = st.session_state.data
        detected_encoding = st.session_state.last_successful_encoding

    if st.session_state.data is None:
        st.stop()

    if st.session_state.data_load_message:
        st.success(st.session_state.data_load_message)

    # Prepare data structure
    intents = {}
    for col in df.columns:
        phrases = df[col].dropna().tolist()
        # Remove empty strings and strip whitespace
        phrases = [str(p).strip() for p in phrases if str(p).strip() and str(p).strip().lower() != 'nan']
        if phrases:  # Only add if there are actual phrases
            intents[col] = phrases

    # Show dataframe summary
    st.info(f"Total phrases across all intents: {sum(len(p) for p in intents.values())}")

    # TAB 1: SIMILARITY ANALYSIS
    with tab1:
        st.header("📊 Intent Data")
        st.dataframe(df, width='stretch')

        # NOW show verification (after intents exists)
        if intents:  # Only show if intents dictionary has data
            with st.expander("🔍 Verify Data Structure", expanded=False):
                st.write("**First 3 phrases from each intent:**")
                for intent_name, phrases_list in list(intents.items())[:5]:  # Show first 5 intents
                    st.write(f"**{intent_name}:** ({len(phrases_list)} total phrases)")
                    for idx, phrase in enumerate(phrases_list[:3], 1):
                        st.write(f"  {idx}. {phrase[:100]}..." if len(phrase) > 100 else f"  {idx}. {phrase}")
                    if len(phrases_list) > 3:
                        st.write(f"  ... and {len(phrases_list) - 3} more")

        # Check for potential issues
        all_phrases_check = []
        for phrases_list in intents.values():
            all_phrases_check.extend(phrases_list)

        duplicates = len(all_phrases_check) - len(set(all_phrases_check))
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate phrases across intents. This may inflate similarity scores.")

            # Find and show duplicates
            with st.expander("🔍 View Duplicate Phrases", expanded=False):
                phrase_counts = Counter(all_phrases_check)
                duplicate_phrases = {phrase: count for phrase, count in phrase_counts.items() if count > 1}

                if duplicate_phrases:
                    st.write(f"**Found {len(duplicate_phrases)} unique phrases that appear multiple times:**")

                    dup_data = []
                    for phrase, count in sorted(duplicate_phrases.items(), key=lambda x: x[1], reverse=True)[:50]:
                        # Find which intents this phrase appears in
                        appears_in = []
                        for intent_name, phrases_list in intents.items():
                            if phrase in phrases_list:
                                appears_in.append(intent_name)

                        dup_data.append({
                            'Phrase': phrase[:80] + '...' if len(phrase) > 80 else phrase,
                            'Count': count,
                            'Appears In': ', '.join(appears_in)
                        })

                    dup_df = pd.DataFrame(dup_data)
                    st.dataframe(dup_df, width='stretch')

                    # Option to remove duplicates
                    if st.button("🧹 Remove Duplicates from Dataset"):
                        cleaned_df = df.copy()
                        for col in cleaned_df.columns:
                            # Remove duplicates within each column
                            cleaned_df[col] = cleaned_df[col].drop_duplicates()

                        # Remove rows that are all NaN
                        cleaned_df = cleaned_df.dropna(how='all')

                        st.session_state.data = cleaned_df
                        st.session_state.analysis_results = None
                        st.session_state.embeddings = None
                        st.success(f"✅ Removed duplicates! Re-run analysis to see improvements.")
                        st.download_button(
                            "📥 Download Cleaned Dataset",
                            cleaned_df.to_csv(index=False),
                            "cleaned_intents.csv",
                            "text/csv"
                        )

        # Large dataset warning
        total_phrases = sum(len(p) for p in intents.values())
        if total_phrases > 1000:
            st.warning(f"⚠️ Large dataset detected ({total_phrases} phrases). Analysis will use smart sampling to prevent browser crashes. Full results available via downloads.")
            st.info("💡 **Pro Tip**: For large datasets, increase the similarity threshold filter to focus on the most critical issues first.")

        # Analysis section
        if st.session_state.model is not None:
            st.header("🔍 Similarity Analysis")

            # Run analysis button
            run_analysis_requested = st.button("Run Analysis", key="run_analysis_tab1")
            if run_analysis_requested:
                current_device = get_device()
                with st.spinner("Computing embeddings..."):
                    analysis_results = perform_analysis(
                        intents,
                        st.session_state.model,
                        st.session_state.tokenizer,
                        st.session_state.model_type,
                        device_obj=current_device,
                        embed_batch_size=st.session_state.embedding_batch_size,
                        phrase_chunk_size=st.session_state.phrase_similarity_chunk_size,
                        use_mixed_precision=st.session_state.use_mixed_precision and current_device.type == 'cuda',
                        show_progress=True,
                        pooling_strategy=st.session_state.pooling_strategy
                    )
                    st.session_state.analysis_results = analysis_results
                    st.session_state.embeddings = analysis_results['embeddings']
                    st.success("✅ Analysis complete!")

            # Display results if available
            if st.session_state.analysis_results:
                analysis_results = st.session_state.analysis_results

                # Extract results
                all_phrases = analysis_results['all_phrases']
                phrase_to_intent = analysis_results['phrase_to_intent']
                embeddings = analysis_results['embeddings']
                intent_names = analysis_results['intent_names']
                intent_sim_matrix = analysis_results['intent_sim_matrix']

                # Find confusing pairs
                confusing_pairs = []
                for i in range(len(intent_names)):
                    for j in range(i + 1, len(intent_names)):
                        sim = intent_sim_matrix[i][j]
                        if sim >= similarity_threshold:
                            confusing_pairs.append({
                                'Intent A': intent_names[i],
                                'Intent B': intent_names[j],
                                'Similarity': f"{sim:.3f}",
                                'Risk': '🔴 High' if sim > 0.9 else '🟡 Medium'
                            })

                # Display summary
                st.subheader("📋 Executive Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Intents", len(intent_names))

                with col2:
                    st.metric("Total Phrases", len(all_phrases))

                with col3:
                    st.metric("Unique Phrases", len(set(all_phrases)))

                with col4:
                    st.metric("Confusing Pairs", len(confusing_pairs))

                # Display heatmap
                st.subheader("🎯 Intent Similarity Matrix")
                fig = px.imshow(
                    intent_sim_matrix,
                    x=intent_names,
                    y=intent_names,
                    color_continuous_scale='RdYlGn_r',
                    labels=dict(color="Similarity"),
                    title="Intent Similarity Matrix"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, width='stretch')

                # Display confusing pairs
                if confusing_pairs:
                    st.subheader("⚠️ Potentially Confusing Intent Pairs")
                    st.dataframe(pd.DataFrame(confusing_pairs), width='stretch')
                else:
                    st.success("✅ No confusing intent pairs detected at this threshold!")

                # Phrase-level analysis
                st.subheader("📝 Cross-Intent Phrase Confusion")

                # Filter phrase conflicts
                phrase_conflicts_all = analysis_results['phrase_conflicts']
                phrase_conflict_limit_reached = analysis_results['phrase_conflict_limit_reached']

                # Add filtering options
                col1, col2, col3 = st.columns(3)

                with col1:
                    min_similarity_filter = st.slider(
                        "Show only phrases above similarity:",
                        min_value=PHRASE_SIMILARITY_MIN,
                        max_value=1.0,
                        value=max(phrase_similarity_threshold, PHRASE_SIMILARITY_MIN),
                        step=0.05,
                        help="Filter to show only the most critical phrase conflicts",
                        key="tab1_phrase_filter"
                    )

                with col2:
                    max_display_rows = st.selectbox(
                        "Max rows to display:",
                        [50, 100, 500, 1000],
                        index=1,
                        help="Limit rows shown in UI to prevent browser overload",
                        key="tab1_max_rows"
                    )

                with col3:
                    specific_intent_filter = st.selectbox(
                        "Filter by intent:",
                        ["All Intents"] + list(intents.keys()),
                        help="Focus on a specific intent",
                        key="tab1_intent_filter"
                    )

                # Filter precomputed phrase conflicts
                st.info("🔍 Filtering phrase-level confusion results...")
                if phrase_conflict_limit_reached:
                    st.warning(f"⚠️ Phrase conflict analysis capped at {PHRASE_CONFLICT_MAX:,} results (based on your system's {DYNAMIC_LIMITS['available_gb']}GB available memory - {DYNAMIC_LIMITS['memory_tier']} tier).")

                phrase_confusion = []
                for conflict in phrase_conflicts_all:
                    if conflict['Similarity'] < min_similarity_filter:
                        continue
                    if specific_intent_filter != "All Intents":
                        if conflict['Intent'] != specific_intent_filter and conflict['Other Intent'] != specific_intent_filter:
                            continue
                    phrase_confusion.append(conflict.copy())

                if phrase_confusion:
                    st.warning(f"Found {len(phrase_confusion)} phrase conflicts above {min_similarity_filter:.2f} similarity")

                    # Create dataframe
                    confusion_df = pd.DataFrame(phrase_confusion)

                    # Separate exact duplicates from similar phrases
                    if 'Note' in confusion_df.columns:
                        exact_dupes = confusion_df[confusion_df['Note'] == '⚠️ EXACT DUPLICATE']
                        similar_phrases = confusion_df[confusion_df['Note'] != '⚠️ EXACT DUPLICATE']

                        if len(exact_dupes) > 0:
                            st.error(f"🔴 **CRITICAL DATA ISSUE**: Found {len(exact_dupes)} exact duplicate phrases across different intents!")

                            with st.expander(f"🔍 View ALL {len(exact_dupes)} Exact Duplicates", expanded=False):
                                st.dataframe(exact_dupes, width='stretch', height=400)
                                # Download exact duplicates
                                csv_dupes = exact_dupes.to_csv(index=False)
                                st.download_button(
                                    f"📥 Download ALL Exact Duplicates ({len(exact_dupes)} rows)",
                                    csv_dupes,
                                    "exact_duplicates.csv",
                                    "text/csv",
                                    key='download-exact-dupes-tab1'
                                )

                        st.info(f"Found {len(similar_phrases)} similar (but not identical) phrase pairs")
                        confusion_df = similar_phrases

                    # Sort by similarity
                    confusion_df['Similarity_Float'] = confusion_df['Similarity'].astype(float)
                    confusion_df = confusion_df.sort_values('Similarity_Float', ascending=False)
                    confusion_df = confusion_df.drop('Similarity_Float', axis=1)

                    # Show limited rows
                    display_df = confusion_df.head(max_display_rows)
                    st.dataframe(display_df, width='stretch')

                    if len(confusion_df) > max_display_rows:
                        st.warning(f"⚠️ {len(confusion_df) - max_display_rows} additional conflicts not shown.")

                    # Download button
                    csv_confusion = confusion_df.to_csv(index=False)
                    st.download_button(
                        f"📥 Download Full Confusion Report ({len(confusion_df)} rows)",
                        csv_confusion,
                        "phrase_confusion_report.csv",
                        "text/csv",
                        key='download-confusion-tab1'
                    )
                else:
                    st.success(f"✅ No phrase confusion detected above {min_similarity_filter:.2f} similarity!")

                # RECOMMENDATIONS SECTION
                st.header("💡 Actionable Recommendations")

                # Quick Action Summary Card
                with st.container():
                    st.markdown("### ⚡ Quick Action Summary")

                    critical_pairs = len([p for p in confusing_pairs if float(p['Similarity']) > 0.92])
                    high_sim_pairs = len([p for p in confusing_pairs if 0.85 <= float(p['Similarity']) <= 0.92])

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("🔴 Must Fix", critical_pairs)

                    with col2:
                        st.metric("🟡 Should Fix", high_sim_pairs)

                    with col3:
                        phrase_critical = len([p for p in phrase_confusion[:1000] if float(p['Similarity']) > 0.95]) if phrase_confusion else 0
                        st.metric("📝 Phrases to Move/Delete", f"{phrase_critical}+")

                    with col4:
                        total_conflicts = len(phrase_confusion) if phrase_confusion else 0
                        # Show "+" if at limit, otherwise exact count
                        if phrase_conflict_limit_reached and total_conflicts >= PHRASE_CONFLICT_MAX:
                            display_text = f"{PHRASE_CONFLICT_MAX:,}+"
                        else:
                            display_text = f"{total_conflicts:,}"
                        st.metric("📊 Total Conflicts", display_text)

                    st.markdown("---")

                # Save baseline for comparison
                if st.session_state.baseline_results is None:
                    if st.button("📊 Set as Baseline (for before/after comparison)", key="set_baseline_tab1"):
                        st.session_state.baseline_results = {
                            'confusing_pairs': confusing_pairs,
                            'phrase_confusion': phrase_confusion,
                            'intent_sim_matrix': intent_sim_matrix
                        }
                        st.success("✅ Baseline saved! Make changes and re-run to see improvements.")
                else:
                    st.info("📊 Baseline comparison available - see below")

                # PRIORITY RECOMMENDATIONS
                st.subheader("🎯 Priority Actions (Ranked by Impact)")

                priority_actions = []

                # High-priority: Pairs above 0.9 similarity
                current_device = get_device()
                for pair in confusing_pairs:
                    sim_score = float(pair['Similarity'])
                    if sim_score > 0.9:
                        intent_a = pair['Intent A']
                        intent_b = pair['Intent B']

                        # Analyze keyword overlap
                        keyword_analysis = analyze_keyword_overlap(
                            intents[intent_a],
                            intents[intent_b]
                        )

                        priority_actions.append({
                            'Priority': '🔴 CRITICAL',
                            'Action': f"Merge or significantly differentiate '{intent_a}' and '{intent_b}'",
                            'Reason': f"Extremely high similarity ({sim_score:.3f})",
                            'Impact': 'High',
                            'Common Keywords': ', '.join(keyword_analysis['common_keywords'][:5]) if keyword_analysis['common_keywords'] else 'None',
                            'Suggested Fix': f"Add unique keywords to each. For '{intent_a}': {', '.join(keyword_analysis['intent1_unique'][:3])}. For '{intent_b}': {', '.join(keyword_analysis['intent2_unique'][:3])}"
                        })

                # Medium-priority: Specific phrase confusions
                if phrase_confusion:
                    phrase_issues = {}
                    sample_conflicts = phrase_confusion[:1000] if len(phrase_confusion) > 1000 else phrase_confusion

                    for confusion in sample_conflicts:
                        key = (confusion['Intent'], confusion['Other Intent'])
                        if key not in phrase_issues:
                            phrase_issues[key] = []
                        phrase_issues[key].append(confusion)

                    for (intent_a, intent_b), confusions in sorted(phrase_issues.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
                        priority_actions.append({
                            'Priority': '🟡 HIGH',
                            'Action': f"Review {len(confusions)} phrases in '{intent_a}' similar to '{intent_b}'",
                            'Reason': f"{len(confusions)} phrases causing confusion",
                            'Impact': 'Medium',
                            'Common Keywords': 'See phrase-level details',
                            'Suggested Fix': f"Rephrase or move the most similar phrases"
                        })

                if priority_actions:
                    priority_df = pd.DataFrame(priority_actions)
                    st.dataframe(priority_df, width='stretch', height=400)

                    csv_priority = priority_df.to_csv(index=False)
                    st.download_button(
                        f"📥 Download All Priority Actions ({len(priority_df)} items)",
                        csv_priority,
                        "priority_actions.csv",
                        "text/csv",
                        key='download-priority-tab1'
                    )
                else:
                    st.success("✅ No priority actions needed!")

                st.markdown("---")

                # PHRASE-LEVEL RECOMMENDATIONS
                st.subheader("📝 Phrase-Level Actions")

                if phrase_confusion:
                    st.info(f"💡 Analyzing top {min(len(phrase_confusion), 500)} phrase conflicts for actionable recommendations...")

                    # Group and prioritize phrase-level issues (limit for performance)
                    phrase_recommendations = []

                    # Only analyze top N most similar for recommendations
                    sample_size = min(len(phrase_confusion), 500)

                    for confusion in phrase_confusion[:sample_size]:
                        phrase = confusion['Phrase']
                        current_intent = confusion['Intent']
                        similar_phrase = confusion['Similar To']
                        other_intent = confusion['Other Intent']
                        similarity = float(confusion['Similarity'])

                        # Determine action based on similarity
                        if similarity > 0.95:
                            action = "🔴 MOVE or DELETE"
                            reason = "Almost identical to other intent"
                        elif similarity > 0.9:
                            action = "🟡 REPHRASE"
                            reason = "Very similar - needs differentiation"
                        else:
                            action = "🟢 REVIEW"
                            reason = "Moderately similar - consider rewording"

                        # Extract keywords from both phrases
                        keywords_phrase = extract_keywords(phrase)
                        keywords_similar = extract_keywords(similar_phrase)
                        common = set(keywords_phrase) & set(keywords_similar)
                        unique_to_current = set(keywords_phrase) - common
                        unique_to_other = set(keywords_similar) - common

                        phrase_recommendations.append({
                            'Action': action,
                            'Your Phrase': phrase[:80] + '...' if len(phrase) > 80 else phrase,
                            'Current Intent': current_intent,
                            'Conflicts With Phrase': similar_phrase[:80] + '...' if len(similar_phrase) > 80 else similar_phrase,
                            'Other Intent': other_intent,
                            'Similarity': f"{similarity:.3f}",
                            'Reason': reason,
                            'Common Words': ', '.join(list(common)[:5]) if common else 'None',
                            'Add to Current': ', '.join(list(unique_to_other)[:3]) if unique_to_other else 'N/A',
                            'Suggestion': f"Either move to '{other_intent}', delete, or add unique words like: {', '.join(list(unique_to_other)[:3])}" if unique_to_other else "Consider removing one of these phrases"
                        })

                    # Sort by similarity (highest first) and maintain columns even when empty
                    phrase_columns = [
                        'Action',
                        'Your Phrase',
                        'Current Intent',
                        'Conflicts With Phrase',
                        'Other Intent',
                        'Similarity',
                        'Reason',
                        'Common Words',
                        'Add to Current',
                        'Suggestion'
                    ]
                    phrase_df = pd.DataFrame(phrase_recommendations, columns=phrase_columns)
                    if not phrase_df.empty:
                        phrase_df['Sim_Sort'] = phrase_df['Similarity'].astype(float)
                        phrase_df = phrase_df.sort_values('Sim_Sort', ascending=False)
                        phrase_df = phrase_df.drop('Sim_Sort', axis=1)

                    # Display controls
                    col1, col2 = st.columns(2)
                    with col1:
                        action_filter = st.multiselect(
                            "Filter by action type:",
                            ["🔴 MOVE or DELETE", "🟡 REPHRASE", "🟢 REVIEW"],
                            default=["🔴 MOVE or DELETE", "🟡 REPHRASE"],
                            key="action_filter_tab1"
                        )

                    with col2:
                        display_limit = st.selectbox(
                            "Rows to display:",
                            [20, 50, 250, 2000],
                            index=1,
                            key="phrase_display_limit_tab1"
                        )

                    # Apply filters
                    if action_filter:
                        filtered_df = phrase_df[phrase_df['Action'].isin(action_filter)]
                    else:
                        filtered_df = phrase_df

                    # Display limited rows with both phrases visible
                    display_df = filtered_df.head(display_limit)
                    if display_df.empty:
                        st.warning("No phrase-level actions match the current filters.")
                    else:
                        st.dataframe(display_df, width='stretch', height=400)
                        st.info("💡 **How to read this table:** 'Your Phrase' is the problem phrase. 'Conflicts With Phrase' is the similar phrase in the other intent. Compare them side-by-side to decide what to do.")
                        if len(filtered_df) > display_limit:
                            st.info(f"📊 Showing {display_limit} of {len(filtered_df)} recommendations. Download full list below.")

                    # Download phrase-level recommendations
                    csv_phrases = phrase_df.to_csv(index=False)
                    st.download_button(
                        f"📥 Download All Phrase-Level Actions ({len(phrase_df)} rows)",
                        csv_phrases,
                        "phrase_level_actions.csv",
                        "text/csv",
                        key='download-phrase-actions-tab1',
                        help="CSV includes BOTH conflicting phrases for easy comparison"
                    )

                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        critical_count = len(phrase_df[phrase_df['Action'] == '🔴 MOVE or DELETE'])
                        st.metric("🔴 Critical Actions", critical_count)
                    with col2:
                        rephrase_count = len(phrase_df[phrase_df['Action'] == '🟡 REPHRASE'])
                        st.metric("🟡 Rephrase Needed", rephrase_count)
                    with col3:
                        review_count = len(phrase_df[phrase_df['Action'] == '🟢 REVIEW'])
                        st.metric("🟢 Review", review_count)

                else:
                    st.success("✅ No phrase-level conflicts detected!")

                st.markdown("---")

                # KEYWORD ANALYSIS
                st.subheader("🏷️ Keyword Overlap Analysis")

                if confusing_pairs:
                    st.write("**Why are these intents confusing? Common keywords:**")

                    for pair in confusing_pairs[:5]:
                        intent_a = pair['Intent A']
                        intent_b = pair['Intent B']

                        keyword_analysis = analyze_keyword_overlap(
                            intents[intent_a],
                            intents[intent_b]
                        )

                        with st.expander(f"📊 {intent_a} ↔️ {intent_b} (Similarity: {pair['Similarity']})"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown(f"**Common Keywords ({len(keyword_analysis['common_keywords'])}):**")
                                if keyword_analysis['common_keywords']:
                                    for kw in keyword_analysis['common_keywords'][:10]:
                                        st.markdown(f"- `{kw}`")
                                else:
                                    st.write("None found")

                            with col2:
                                st.markdown(f"**Unique to '{intent_a}':**")
                                if keyword_analysis['intent1_unique']:
                                    for kw in keyword_analysis['intent1_unique'][:10]:
                                        st.markdown(f"- `{kw}`")
                                else:
                                    st.write("None found")

                            with col3:
                                st.markdown(f"**Unique to '{intent_b}':**")
                                if keyword_analysis['intent2_unique']:
                                    for kw in keyword_analysis['intent2_unique'][:10]:
                                        st.markdown(f"- `{kw}`")
                                else:
                                    st.write("None found")

                st.markdown("---")

                # MERGE VS DIFFERENTIATE GUIDANCE
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔀 Should You Merge?")
                    merge_candidates = [p for p in confusing_pairs if float(p['Similarity']) > 0.92]
                    if merge_candidates:
                        st.warning(f"Found {len(merge_candidates)} intent pair(s) that might be duplicates:")
                        for pair in merge_candidates[:3]:
                            st.markdown(f"- **{pair['Intent A']}** ↔️ **{pair['Intent B']}** ({pair['Similarity']})")
                        st.info("✅ **Consider merging** if these intents serve the same business purpose")
                    else:
                        st.success("No clear merge candidates!")

                with col2:
                    st.subheader("📊 How to Differentiate")
                    differentiate_candidates = [p for p in confusing_pairs if 0.85 <= float(p['Similarity']) <= 0.92]
                    if differentiate_candidates:
                        st.info(f"Found {len(differentiate_candidates)} intent pair(s) that need clearer separation:")
                        for pair in differentiate_candidates[:3]:
                            st.markdown(f"- **{pair['Intent A']}** ↔️ **{pair['Intent B']}** ({pair['Similarity']})")
                        st.info("✅ **Add distinctive phrases** using unique keywords")
                    else:
                        st.success("All distinct intents are well-separated!")

                st.markdown("---")

                # BEFORE/AFTER COMPARISON
                if st.session_state.baseline_results is not None:
                    st.subheader("📈 Before/After Comparison")

                    baseline = st.session_state.baseline_results

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Confusing Pairs",
                            len(confusing_pairs),
                            delta=len(confusing_pairs) - len(baseline['confusing_pairs']),
                            delta_color="inverse"
                        )

                    with col2:
                        st.metric(
                            "Phrase Conflicts",
                            len(phrase_confusion),
                            delta=len(phrase_confusion) - len(baseline['phrase_confusion']),
                            delta_color="inverse"
                        )

                    with col3:
                        current_avg = np.mean([float(p['Similarity']) for p in confusing_pairs]) if confusing_pairs else 0
                        baseline_avg = np.mean([float(p['Similarity']) for p in baseline['confusing_pairs']]) if baseline['confusing_pairs'] else 0
                        st.metric(
                            "Avg Confusion Score",
                            f"{current_avg:.3f}",
                            delta=f"{current_avg - baseline_avg:.3f}",
                            delta_color="inverse"
                        )

                    if len(confusing_pairs) < len(baseline['confusing_pairs']):
                        st.success("🎉 Great progress! You've reduced confusion between intents.")
                    elif len(confusing_pairs) > len(baseline['confusing_pairs']):
                        st.warning("⚠️ Confusion increased. Review recent changes.")
                    else:
                        st.info("No change from baseline.")

                    if st.button("🔄 Reset Baseline", key="reset_baseline_tab1"):
                        st.session_state.baseline_results = None
                        st.rerun()

                # Phrase Generation
                st.header("✨ Generate Improved Training Phrases")

                st.info("💡 Generate new phrases to increase diversity and reduce confusion")

                col1, col2, col3 = st.columns(3)

                with col1:
                    selected_intent = st.selectbox("Select Intent to Generate Phrases For:", list(intents.keys()), key="gen_intent_tab1")

                with col2:
                    generation_strategy = st.selectbox(
                        "Generation Strategy:",
                        ["Diverse Variations", "Targeted Differentiation"],
                        help="Diverse: general variations. Targeted: emphasize unique keywords",
                        key="gen_strategy_tab1"
                    )

                with col3:
                    # Paraphrase model selection
                    paraphrase_model_options = {
                        "humarin/chatgpt_paraphraser_on_T5_base": "ChatGPT-style (Fast, Creative)",
                        "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality": "T5-Large (High Quality, Slower)",
                        "Vamsi/T5_Paraphrase_Paws": "PAWS (Original, May Have Issues)"
                    }

                    selected_paraphrase_model = st.selectbox(
                        "Paraphrase Model:",
                        list(paraphrase_model_options.keys()),
                        format_func=lambda x: paraphrase_model_options[x],
                        help="Choose which AI model to use for generating paraphrases",
                        key="paraphrase_model_tab1"
                    )

                num_generations = st.slider("Number of variations per phrase:", 1, 5, 2, key="num_gen_tab1")

                # Model info
                with st.expander("📘 Paraphrase Model Guide", expanded=False):
                    st.markdown("""
                    ### Available Models:

                    **ChatGPT-style Paraphraser** (Recommended)
                    - ✅ Fast and reliable
                    - ✅ Creative variations
                    - ✅ Good tokenizer compatibility
                    - Size: ~900MB
                    - Best for: Quick generation, modern style

                    **T5-Large Paraphraser**
                    - ✅ High quality outputs
                    - ✅ Diverse variations
                    - ⚠️ Slower (larger model)
                    - Size: ~3GB
                    - Best for: When quality is critical

                    **PAWS (Original)**
                    - ⚠️ May have tokenizer issues
                    - ✅ Good for specific use cases
                    - Size: ~900MB
                    - Best for: Legacy compatibility

                    **Tips:**
                    - First time: Model will download automatically
                    - GPU: Significantly faster if available
                    - Memory: T5-Large needs more RAM
                    """)

                # Show unique keywords for targeted generation
                if generation_strategy == "Targeted Differentiation":
                    confused_with = []
                    for pair in confusing_pairs:
                        if pair['Intent A'] == selected_intent:
                            confused_with.append(pair['Intent B'])
                        elif pair['Intent B'] == selected_intent:
                            confused_with.append(pair['Intent A'])

                    if confused_with:
                        st.warning(f"⚠️ '{selected_intent}' is confused with: {', '.join(confused_with)}")

                        all_unique_keywords = set()
                        for confused_intent in confused_with:
                            keyword_analysis = analyze_keyword_overlap(
                                intents[selected_intent],
                                intents[confused_intent]
                            )
                            all_unique_keywords.update(keyword_analysis['intent1_unique'][:5])

                        if all_unique_keywords:
                            st.info(f"💡 Try to include these unique keywords: **{', '.join(list(all_unique_keywords)[:10])}**")
                    else:
                        st.success(f"✅ '{selected_intent}' has no confusion - generating diverse variations")

                generated_state = st.session_state.generated_phrases.get(selected_intent)

                if st.button("Generate Phrases", key="generate_phrases_tab1"):
                    # Check for required dependencies first
                    try:
                        import sentencepiece
                    except ImportError:
                        st.error("❌ Missing required dependency: `sentencepiece`")
                        st.markdown("""
                        **To fix this error, install sentencepiece:**

                        ```bash
                        pip install sentencepiece protobuf
                        ```

                        Then restart your Streamlit application.
                        """)
                        st.stop()

                    with st.spinner(f"Loading {paraphrase_model_options[selected_paraphrase_model]} and generating phrases..."):
                        try:
                            # Use the selected paraphrase model
                            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

                            # Load with use_fast=False to avoid SentencePiece conversion issues
                            tokenizer = AutoTokenizer.from_pretrained(selected_paraphrase_model, use_fast=False, legacy=False)
                            model = AutoModelForSeq2SeqLM.from_pretrained(selected_paraphrase_model)
                            generator = pipeline(
                                "text2text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                device=0 if torch.cuda.is_available() and not st.session_state.force_cpu else -1
                            )
                            model_used = selected_paraphrase_model

                            st.info(f"✅ Using: {paraphrase_model_options[model_used]}")

                            new_phrases = []
                            # Limit to first 20 phrases to avoid long generation times
                            sample_phrases = intents[selected_intent][:20] if len(intents[selected_intent]) > 20 else intents[selected_intent]

                            for phrase in sample_phrases:
                                for _ in range(num_generations):
                                    # Format input based on model
                                    if "chatgpt_paraphraser" in model_used:
                                        input_text = f"paraphrase: {phrase}"
                                    else:
                                        input_text = f"paraphrase: {phrase} </s>"

                                    result = generator(
                                        input_text,
                                        max_new_tokens=128,
                                        num_return_sequences=1,
                                        do_sample=True,
                                        temperature=0.7
                                    )
                                    generated_text = result[0]['generated_text']
                                    # Clean up the generated text
                                    generated_text = generated_text.replace("paraphrased: ", "").strip()
                                    new_phrases.append(generated_text)

                            if len(intents[selected_intent]) > 20:
                                st.warning(f"⚠️ Generated variations for first 20 phrases only (out of {len(intents[selected_intent])}) to keep generation time reasonable.")

                            st.session_state.generated_phrases[selected_intent] = {
                                'phrases': new_phrases,
                                'strategy': generation_strategy,
                                'num_variations': num_generations,
                                'generated_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'model_used': model_used
                            }
                            generated_state = st.session_state.generated_phrases[selected_intent]
                            st.success(f"✅ Generated {len(new_phrases)} new phrases!")
                        except Exception as e:
                            st.error(f"❌ Error generating phrases: {str(e)}")

                            with st.expander("💡 Troubleshooting Tips", expanded=True):
                                st.markdown("""
                                **Common Issues:**

                                1. **Missing SentencePiece Library** (Most Common):
                                   ```bash
                                   pip install sentencepiece protobuf
                                   ```
                                   Then restart your Streamlit app. This is required for all T5-based models.

                                2. **First Time Running**: The model needs to download (~500MB-3GB depending on model). This may take a few minutes.

                                3. **Internet Connection**: Make sure you have a stable internet connection for the first download.

                                4. **Memory Issues**: Paraphrase generation is memory-intensive. Try:
                                   - Reducing the number of variations per phrase
                                   - Using "Force CPU" if GPU memory is limited
                                   - Closing other applications
                                   - Choosing ChatGPT-style instead of T5-Large

                                5. **Still Having Issues?**: Try upgrading all dependencies:
                                   ```bash
                                   pip install --upgrade transformers sentencepiece protobuf torch
                                   ```

                                6. **Alternative**: Use the semantic search feature in Tab 2 to find similar phrases
                                   from your existing dataset instead of generating new ones.
                                """)

                            st.session_state.generated_phrases.pop(selected_intent, None)
                            generated_state = None

                generated_state = st.session_state.generated_phrases.get(selected_intent)
                if generated_state:
                    phrases_for_display = generated_state.get('phrases', [])
                    st.caption(f"Last generated: {generated_state.get('generated_at', 'unknown time')}")
                    gen_df = pd.DataFrame({selected_intent: phrases_for_display})
                    st.dataframe(gen_df, width='stretch')

                    col_add, col_clear = st.columns(2)
                    with col_add:
                        if st.button("Add Generated Phrases to Dataset", key="add_gen_tab1"):
                            if phrases_for_display:
                                extended_df = df.copy()
                                for phrase in phrases_for_display:
                                    new_row = {col: None for col in extended_df.columns}
                                    new_row[selected_intent] = phrase
                                    extended_df = pd.concat([extended_df, pd.DataFrame([new_row])], ignore_index=True)
                                st.session_state.data = extended_df
                                st.success("Phrases added!")
                                st.session_state.generated_phrases.pop(selected_intent, None)
                                st.session_state.analysis_results = None
                                st.session_state.embeddings = None
                                st.rerun()
                    with col_clear:
                        if st.button("Clear Generated Phrases", key="clear_gen_tab1"):
                            st.session_state.generated_phrases.pop(selected_intent, None)
                            st.rerun()

                # Export section
                st.header("💾 Export Data & Next Steps")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Generate Excel workbook
                    if st.button("📊 Generate Complete Excel Workbook", type="primary", key="excel_tab1"):
                        with st.spinner("Creating Excel workbook..."):
                            try:
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    # Tab 1: Executive Summary
                                    summary_data = {
                                        'Metric': [
                                            'Total Intents',
                                            'Total Phrases',
                                            'Unique Phrases',
                                            'Confusing Intent Pairs',
                                            'Total Phrase Conflicts',
                                            'Intent Threshold Used',
                                            'Phrase Threshold Used'
                                        ],
                                        'Value': [
                                            len(intent_names),
                                            sum(len(p) for p in intents.values()),
                                            len(set(all_phrases)),
                                            len(confusing_pairs),
                                            len(phrase_confusion) if phrase_confusion else 0,
                                            similarity_threshold,
                                            phrase_similarity_threshold
                                        ]
                                    }
                                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                                    # Tab 2: Confusing Intent Pairs
                                    if confusing_pairs:
                                        pd.DataFrame(confusing_pairs).to_excel(writer, sheet_name='Confusing_Intent_Pairs', index=False)

                                    # Tab 3: Priority Actions
                                    if priority_actions:
                                        pd.DataFrame(priority_actions).to_excel(writer, sheet_name='Priority_Actions', index=False)

                                    # Tab 4: All Phrase Conflicts
                                    if phrase_confusion:
                                        pd.DataFrame(phrase_confusion).to_excel(writer, sheet_name='All_Phrase_Conflicts', index=False)

                                    # Tab 5: Intent Similarity Matrix
                                    intent_sim_df = pd.DataFrame(
                                        intent_sim_matrix,
                                        columns=intent_names,
                                        index=intent_names
                                    )
                                    intent_sim_df.to_excel(writer, sheet_name='Intent_Similarity_Matrix')

                                output.seek(0)

                                st.download_button(
                                    "📥 Download Complete Excel Workbook",
                                    output,
                                    "intent_analysis_complete.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key='download-excel-tab1'
                                )
                                st.success("✅ Excel workbook generated!")
                            except Exception as e:
                                st.error(f"Error creating Excel file: {str(e)}")

                with col2:
                    csv = st.session_state.data.to_csv(index=False)
                    st.download_button(
                        "📥 Download Modified Dataset",
                        csv,
                        "modified_intents.csv",
                        "text/csv",
                        key='download-csv-tab1'
                    )

                with col3:
                    if st.button("📄 Generate Full Report", key="report_tab1"):
                        report = f"""# Intent Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Intents: {len(intent_names)}
- Total Phrases: {sum(len(p) for p in intents.values())}
- Confusing Intent Pairs: {len(confusing_pairs)}
- Total Phrase Conflicts: {len(phrase_confusion) if phrase_confusion else 0}

## Top Confusing Intent Pairs
"""
                        for pair in confusing_pairs[:10]:
                            report += f"\n- {pair['Intent A']} ↔️ {pair['Intent B']}: {pair['Similarity']}\n"

                        report += """
## Recommended Next Steps
1. Address CRITICAL priority items immediately
2. Review HIGH priority phrase-level conflicts
3. Add unique keywords from keyword analysis
4. Generate diverse phrases for low-diversity intents
5. Re-run analysis to validate improvements
"""

                        st.download_button(
                            "📥 Download Full Report",
                            report,
                            "intent_analysis_report.txt",
                            "text/plain",
                            key='download-report-tab1'
                        )

                # Action Checklist
                st.markdown("---")
                st.subheader("✅ Next Steps Checklist")

                checklist = []
                if critical_pairs > 0:
                    checklist.append("🔴 Address CRITICAL priority actions")
                if high_sim_pairs > 0:
                    checklist.append("🟡 Review HIGH priority phrase-level conflicts")
                if phrase_confusion:
                    checklist.append("📝 Implement phrase-level actions")
                if len(confusing_pairs) > 0:
                    checklist.append("🏷️ Emphasize unique keywords")
                checklist.append("✨ Generate new diverse phrases")
                checklist.append("💾 Export modified dataset")
                checklist.append("🔄 Re-run analysis to verify improvements")

                for item in checklist:
                    st.markdown(f"- [ ] {item}")

            else:
                st.info("👆 Click **Run Analysis** to compute similarities")
        else:
            st.warning("👈 Please load the model from the sidebar first")

    # TAB 2: SEMANTIC SEARCH & AUDIT
    with tab2:
        st.header("🔍 Semantic Search & Dataset Audit")
        st.info("Search for semantically similar phrases to audit your dataset or determine where to add new phrases")

        if st.session_state.model is not None and st.session_state.analysis_results is not None:
            analysis_results = st.session_state.analysis_results
            all_phrases = analysis_results['all_phrases']
            phrase_to_intent = analysis_results['phrase_to_intent']
            embeddings = analysis_results['embeddings']
            intent_names = analysis_results['intent_names']

            # Search interface with two columns layout
            col1, col2 = st.columns([2, 1])

            with col1:
                search_query = st.text_area(
                    "Enter phrase to search or audit:",
                    placeholder="e.g., 'I want to book a flight to Paris' or 'Cancel my subscription'",
                    height=100,
                    help="Enter any phrase to find semantically similar phrases across all intents. Use this to audit existing phrases or determine where new phrases should be added."
                )

                # Add quick examples
                with st.expander("📝 Example Use Cases", expanded=False):
                    st.markdown("""
                    **Dataset Auditing:**
                    - Search for a phrase already in your dataset to find potential duplicates
                    - Check if similar phrases are incorrectly split across multiple intents

                    **Adding New Phrases:**
                    - Enter a new phrase to see which intent it's most similar to
                    - Check similarity scores to determine if it fits well with existing phrases
                    - If similarity is low (<0.7), consider creating a new intent

                    **Quality Check:**
                    - Search for edge cases to see how they match
                    - Verify that similar phrases are grouped in the same intent
                    """)

            with col2:
                st.markdown("### Search Settings")
                search_threshold = st.slider(
                    "Min similarity score:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Set to 0 to see all phrases ranked by similarity"
                )

                top_k = st.number_input(
                    "Max results to show:",
                    min_value=10,
                    max_value=1000,
                    value=50,
                    step=10,
                    help="Maximum number of results to return"
                )

                include_exact = st.checkbox(
                    "Include exact matches",
                    value=True,
                    help="Include phrases that are exactly the same as the search query"
                )

                show_recommendations = st.checkbox(
                    "Show recommendations",
                    value=True,
                    help="Show recommendations for where to add this phrase"
                )

            # Initialize session state for search results persistence
            if 'last_search_results' not in st.session_state:
                st.session_state.last_search_results = None
            if 'last_search_query' not in st.session_state:
                st.session_state.last_search_query = None
            if 'last_search_metadata' not in st.session_state:
                st.session_state.last_search_metadata = {}

            # Search button
            if st.button("🔍 Search & Analyze", type="primary", disabled=not search_query):
                with st.spinner("Searching for semantically similar phrases..."):
                    try:
                        current_device = get_device()

                        # Perform semantic search
                        search_results = semantic_search(
                            search_query,
                            embeddings,
                            all_phrases,
                            phrase_to_intent,
                            st.session_state.model,
                            st.session_state.tokenizer,
                            st.session_state.model_type,
                            current_device,
                            st.session_state.pooling_strategy,
                            threshold=search_threshold,
                            top_k=top_k
                        )

                        # Filter exact matches if needed
                        if not include_exact:
                            search_results = [r for r in search_results if r['phrase'].lower().strip() != search_query.lower().strip()]

                        # Store results in session state
                        st.session_state.last_search_results = search_results
                        st.session_state.last_search_query = search_query
                        st.session_state.last_search_metadata = {
                            'threshold': search_threshold,
                            'top_k': top_k,
                            'include_exact': include_exact,
                            'show_recommendations': show_recommendations
                        }

                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
                        st.info("Make sure the analysis has been run first in the Similarity Analysis tab")
                        st.session_state.last_search_results = None

            # Display results from session state (persists across filter changes)
            search_results = st.session_state.last_search_results
            if search_results is not None:
                search_query = st.session_state.last_search_query
                metadata = st.session_state.last_search_metadata
                search_threshold = metadata.get('threshold', 0.5)
                show_recommendations = metadata.get('show_recommendations', True)

                if search_results:
                            st.success(f"✅ Found {len(search_results)} similar phrases with similarity ≥ {search_threshold:.2f}")

                            # RECOMMENDATIONS SECTION
                            if show_recommendations:
                                st.markdown("---")
                                st.subheader("🎯 Recommendations")

                                # Get top match
                                top_match = search_results[0]
                                top_similarity = top_match['similarity']
                                top_intent = top_match['intent']

                                # Analyze intent distribution in top 10 results
                                top_10_intents = [r['intent'] for r in search_results[:min(10, len(search_results))]]
                                intent_freq = Counter(top_10_intents)
                                dominant_intent = intent_freq.most_common(1)[0][0]
                                dominant_count = intent_freq[dominant_intent]

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("Top Match Similarity", f"{top_similarity:.3f}")
                                    if top_similarity > 0.95:
                                        st.caption("🔴 Very High - Likely duplicate")
                                    elif top_similarity > 0.85:
                                        st.caption("🟡 High - Good fit")
                                    elif top_similarity > 0.70:
                                        st.caption("🟢 Moderate - Acceptable fit")
                                    else:
                                        st.caption("🔵 Low - Consider new intent")

                                with col2:
                                    st.metric("Best Matching Intent", dominant_intent)
                                    st.caption(f"{dominant_count}/10 top matches")

                                with col3:
                                    confidence = dominant_count / min(10, len(search_results))
                                    st.metric("Confidence", f"{confidence:.0%}")
                                    if confidence > 0.7:
                                        st.caption("✅ High confidence")
                                    else:
                                        st.caption("⚠️ Mixed results")

                                # Detailed recommendation
                                st.markdown("#### 💡 Action Recommendation:")

                                if top_similarity > 0.98:
                                    st.error(f"**🔴 DUPLICATE DETECTED:** This phrase is nearly identical to existing phrases in **{top_intent}**. Do not add.")
                                    st.markdown("**Similar phrases:**")
                                    for r in search_results[:3]:
                                        if r['similarity'] > 0.95:
                                            st.markdown(f"- *\"{r['phrase']}\"* (similarity: {r['similarity']:.3f})")

                                elif top_similarity > 0.85 and confidence > 0.7:
                                    st.success(f"**✅ ADD TO INTENT:** Add this phrase to **{dominant_intent}**")
                                    st.markdown(f"**Reasoning:** High similarity ({top_similarity:.3f}) and strong consensus ({confidence:.0%}) indicate this phrase belongs in this intent.")
                                    st.markdown("**Similar phrases in this intent:**")
                                    for r in search_results[:5]:
                                        if r['intent'] == dominant_intent:
                                            st.markdown(f"- *\"{r['phrase']}\"* (similarity: {r['similarity']:.3f})")

                                elif top_similarity > 0.70:
                                    st.warning(f"**🟡 REVIEW NEEDED:** Consider adding to **{dominant_intent}**, but review carefully")
                                    st.markdown(f"**Reasoning:** Moderate similarity ({top_similarity:.3f}) suggests possible fit, but manual review recommended.")

                                    # Check for split across intents
                                    if len(intent_freq) > 1:
                                        st.markdown("**⚠️ Note:** Similar phrases found across multiple intents:")
                                        for intent, count in intent_freq.most_common(3):
                                            st.markdown(f"- {intent}: {count} matches")
                                        st.markdown("This might indicate overlapping intents that need clarification.")

                                else:
                                    st.info(f"**🔵 NEW INTENT SUGGESTED:** Low similarity ({top_similarity:.3f}) to existing phrases")
                                    st.markdown("**Options:**")
                                    st.markdown(f"1. Create a new intent for this type of phrase")
                                    st.markdown(f"2. If related to **{dominant_intent}**, consider adding to expand that intent's scope")
                                    st.markdown(f"3. Review if this phrase is within your chatbot's scope")

                            st.markdown("---")

                            # Prepare results for display with similarity scores
                            results_data = []
                            for result in search_results:
                                similarity = result['similarity']
                                phrase = result['phrase']

                                # Determine match type based on similarity
                                if phrase.lower().strip() == search_query.lower().strip():
                                    match_type = "🎯 Exact"
                                elif similarity > 0.95:
                                    match_type = "🔴 Very High"
                                elif similarity > 0.85:
                                    match_type = "🟡 High"
                                elif similarity > 0.75:
                                    match_type = "🟢 Moderate"
                                elif similarity > 0.60:
                                    match_type = "🔵 Low"
                                else:
                                    match_type = "⚪ Very Low"

                                results_data.append({
                                    'Similarity Score': similarity,  # Keep as float for sorting
                                    'Match Type': match_type,
                                    'Intent': result['intent'],
                                    'Phrase': phrase,
                                    'Keywords': ', '.join(extract_keywords(phrase)[:5])
                                })

                            # Create DataFrame
                            results_df = pd.DataFrame(results_data)

                            # Format similarity score for display
                            results_df['Similarity Score Display'] = results_df['Similarity Score'].apply(lambda x: f"{x:.4f}")

                            # Show statistics
                            st.subheader("📊 Search Statistics")
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                unique_intents = len(results_df['Intent'].unique())
                                st.metric("Unique Intents", unique_intents)

                            with col2:
                                very_high = len(results_df[results_df['Similarity Score'] > 0.95])
                                st.metric("Very Similar (>0.95)", very_high)

                            with col3:
                                high = len(results_df[(results_df['Similarity Score'] > 0.85) & (results_df['Similarity Score'] <= 0.95)])
                                st.metric("High Similar (0.85-0.95)", high)

                            with col4:
                                avg_sim = results_df['Similarity Score'].mean()
                                st.metric("Avg Similarity", f"{avg_sim:.3f}")

                            # Show intent distribution
                            st.subheader("📊 Results by Intent")
                            intent_counts = results_df['Intent'].value_counts()

                            with st.expander("🎯 Intent Distribution Analysis", expanded=True):
                                intent_dist = []
                                for intent, count in intent_counts.items():
                                    intent_results = results_df[results_df['Intent'] == intent]
                                    avg_sim = intent_results['Similarity Score'].mean()
                                    max_sim = intent_results['Similarity Score'].max()
                                    min_sim = intent_results['Similarity Score'].min()
                                    intent_dist.append({
                                        'Intent': intent,
                                        'Count': count,
                                        'Avg Similarity': f"{avg_sim:.3f}",
                                        'Max Similarity': f"{max_sim:.3f}",
                                        'Min Similarity': f"{min_sim:.3f}",
                                        '% of Results': f"{count/len(results_df)*100:.1f}%"
                                    })

                                intent_dist_df = pd.DataFrame(intent_dist)
                                st.dataframe(intent_dist_df, width='stretch')

                            # Show all results
                            st.subheader("📋 All Similar Phrases (Ranked by Similarity)")

                            # Add filtering options
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                filter_intent = st.selectbox(
                                    "Filter by intent:",
                                    ["All"] + list(intent_counts.index),
                                    key="search_filter_intent"
                                )

                            with col2:
                                filter_match_type = st.multiselect(
                                    "Filter by match type:",
                                    ["🎯 Exact", "🔴 Very High", "🟡 High", "🟢 Moderate", "🔵 Low", "⚪ Very Low"],
                                    default=["🎯 Exact", "🔴 Very High", "🟡 High", "🟢 Moderate", "🔵 Low", "⚪ Very Low"],
                                    key="search_filter_match"
                                )

                            with col3:
                                display_limit = st.selectbox(
                                    "Display rows:",
                                    [25, 50, 100, 200, 500],
                                    index=1,
                                    key="search_display_limit"
                                )

                            # Apply filters
                            filtered_results_df = results_df.copy()

                            if filter_intent != "All":
                                filtered_results_df = filtered_results_df[filtered_results_df['Intent'] == filter_intent]

                            if filter_match_type:
                                filtered_results_df = filtered_results_df[filtered_results_df['Match Type'].isin(filter_match_type)]

                            # Display limited results (showing formatted similarity score)
                            display_df = filtered_results_df[['Similarity Score Display', 'Match Type', 'Intent', 'Phrase', 'Keywords']].head(display_limit)
                            display_df = display_df.rename(columns={'Similarity Score Display': 'Similarity Score'})

                            if not display_df.empty:
                                st.dataframe(display_df, width='stretch', height=400)

                                if len(filtered_results_df) > display_limit:
                                    st.info(f"📊 Showing {display_limit} of {len(filtered_results_df)} filtered results")
                            else:
                                st.warning("No results match the current filters")

                            # Download options
                            st.subheader("💾 Export Search Results")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                # CSV download
                                csv_data = results_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download as CSV",
                                    csv_data,
                                    f"semantic_search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv",
                                    key="download_search_csv"
                                )

                            with col2:
                                # Excel download with analysis
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    # Sheet 1: Summary & Recommendations
                                    summary_data = {
                                        'Metric': [
                                            'Search Query',
                                            'Similarity Threshold',
                                            'Total Results',
                                            'Unique Intents',
                                            'Top Match Similarity',
                                            'Top Match Intent',
                                            'Recommended Action',
                                            'Confidence Level'
                                        ],
                                        'Value': [
                                            search_query[:100],
                                            search_threshold,
                                            len(results_df),
                                            unique_intents,
                                            f"{search_results[0]['similarity']:.3f}" if search_results else "N/A",
                                            search_results[0]['intent'] if search_results else "N/A",
                                            "See recommendations" if search_results else "N/A",
                                            f"{confidence:.0%}" if search_results and 'confidence' in locals() else "N/A"
                                        ]
                                    }
                                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                                    # Sheet 2: Intent Distribution
                                    if 'intent_dist_df' in locals():
                                        intent_dist_df.to_excel(writer, sheet_name='Intent_Distribution', index=False)

                                    # Sheet 3: All Results
                                    results_df.to_excel(writer, sheet_name='All_Results', index=False)

                                    # Sheet 4: Top 10 Results for Quick Review
                                    results_df.head(10).to_excel(writer, sheet_name='Top_10_Matches', index=False)

                                output.seek(0)

                                st.download_button(
                                    "📥 Download Excel Report",
                                    output,
                                    f"semantic_search_audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_search_excel"
                                )

                            with col3:
                                # JSON download for programmatic use
                                json_results = {
                                    'query': search_query,
                                    'threshold': search_threshold,
                                    'timestamp': pd.Timestamp.now().isoformat(),
                                    'summary': {
                                        'total_results': len(results_df),
                                        'unique_intents': unique_intents,
                                        'top_similarity': float(search_results[0]['similarity']) if search_results else None,
                                        'recommended_intent': dominant_intent if 'dominant_intent' in locals() else None
                                    },
                                    'results': results_df.to_dict(orient='records')
                                }
                                json_data = json.dumps(json_results, indent=2)
                                st.download_button(
                                    "📥 Download as JSON",
                                    json_data,
                                    f"semantic_search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    "application/json",
                                    key="download_search_json"
                                )

                            # Visualizations
                            st.subheader("📊 Visualizations")

                            # Similarity distribution histogram
                            fig = px.histogram(
                                results_df,
                                x='Similarity Score',
                                nbins=20,
                                title='Distribution of Similarity Scores',
                                labels={'count': 'Number of Phrases', 'Similarity Score': 'Similarity Score'},
                                color_discrete_sequence=['#636EFA']
                            )
                            fig.add_vline(x=0.85, line_dash="dash", line_color="red", annotation_text="High Similarity Threshold")
                            fig.add_vline(x=0.70, line_dash="dash", line_color="yellow", annotation_text="Moderate Threshold")
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, width='stretch')

                            # Intent distribution pie chart
                            if len(intent_counts) > 1:
                                fig2 = px.pie(
                                    values=intent_counts.values,
                                    names=intent_counts.index,
                                    title='Results Distribution by Intent',
                                    height=400
                                )
                                st.plotly_chart(fig2, width='stretch')

                            # Save to search history
                            if st.button("💾 Save to Search History", key="save_search"):
                                st.session_state.search_history.append({
                                    'query': search_query,
                                    'threshold': search_threshold,
                                    'results': len(search_results),
                                    'top_similarity': search_results[0]['similarity'] if search_results else 0,
                                    'recommended_intent': dominant_intent if 'dominant_intent' in locals() else "N/A",
                                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                                st.success("✅ Saved to search history")

                else:
                    st.warning(f"No phrases found with similarity ≥ {search_threshold:.2f}")
                    st.info("💡 Try lowering the similarity threshold to see all results ranked by similarity")

            # Search history
            if st.session_state.search_history:
                st.markdown("---")
                with st.expander("📜 Search History", expanded=False):
                    history_df = pd.DataFrame(st.session_state.search_history)

                    # Display in reverse order (most recent first)
                    history_df = history_df.iloc[::-1]

                    # Format the display
                    display_history = history_df[['timestamp', 'query', 'threshold', 'results', 'top_similarity', 'recommended_intent']].head(20)
                    st.dataframe(display_history, width='stretch')

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📥 Export History", key="export_history"):
                            csv = history_df.to_csv(index=False)
                            st.download_button(
                                "Download History CSV",
                                csv,
                                "search_history.csv",
                                "text/csv",
                                key="download_history"
                            )

                    with col2:
                        if st.button("🗑️ Clear History", key="clear_search_history"):
                            st.session_state.search_history = []
                            st.rerun()

        else:
            if st.session_state.model is None:
                st.warning("👈 Please load the model from the sidebar first")
            else:
                st.warning("👈 Please run the similarity analysis first (in the Similarity Analysis tab)")

else:
    st.info("👈 Upload a CSV file to begin")

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
    - **⚠️ Avoid duplicates** - same phrase shouldn't appear in multiple intents

    ### Features:

    #### 📊 Similarity Analysis Tab
    - Intent-to-intent similarity matrix
    - Phrase-level conflict detection
    - Actionable recommendations
    - Excel/CSV export of results

    #### 🔍 Semantic Search & Audit Tab
    - **Search for semantically similar phrases** with exact similarity scores
    - **Audit your dataset** to find duplicates and misplaced phrases
    - **Determine where to add new phrases** with confidence scores
    - **Get actionable recommendations** based on similarity patterns
    - **Export detailed reports** with similarity scores and analysis
    - **Track search history** for audit trail

    ### Use Cases:

    1. **Dataset Quality Audit**: Search existing phrases to find duplicates or misplaced entries
    2. **Adding New Phrases**: Enter new phrases to see which intent they best match
    3. **Intent Overlap Detection**: Identify when phrases are split across multiple intents
    4. **Coverage Analysis**: Test edge cases to see if they're covered by existing phrases
    """)

# Footer
st.markdown("---")
st.markdown("Built for XLM-RoBERTa Large Intent Classification | Powered by Sentence Transformers")
st.caption("💡 **Note:** Excel export requires openpyxl. Install with: `pip install openpyxl`")
