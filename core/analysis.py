import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import logging

from core.embeddings import encode_texts

logger = logging.getLogger(__name__)

# Try FAISS for fast approximate nearest-neighbor search.
# Falls back to brute-force sklearn cosine_similarity if unavailable.
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not installed -- falling back to brute-force similarity. "
                 "Install faiss-cpu for significantly faster phrase conflict detection.")

PHRASE_SIMILARITY_MIN = 0.70


def _build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product == cosine for L2-normalized vectors
    index.add(embeddings.astype('float32'))
    return index


def _find_phrase_conflicts_faiss(
    embeddings, all_phrases, phrase_to_intent, phrase_conflict_max, k=100
):
    """Use FAISS to find cross-intent phrase conflicts in O(n*k) instead of O(n^2)."""
    index = _build_faiss_index(embeddings)
    n = len(all_phrases)
    k = min(k, n)

    distances, indices = index.search(embeddings.astype('float32'), k)

    conflicts = []
    seen = set()
    for i in range(n):
        for j_pos in range(k):
            j = int(indices[i][j_pos])
            if j <= i:
                continue
            sim = float(distances[i][j_pos])
            if sim < PHRASE_SIMILARITY_MIN:
                continue
            if phrase_to_intent[i] == phrase_to_intent[j]:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            note = '⚠️ EXACT DUPLICATE' if all_phrases[i] == all_phrases[j] else ''
            conflicts.append({
                'Phrase': all_phrases[i],
                'Intent': phrase_to_intent[i],
                'Similar To': all_phrases[j],
                'Other Intent': phrase_to_intent[j],
                'Similarity': sim,
                'Note': note,
            })
            if len(conflicts) >= phrase_conflict_max:
                return conflicts, True
    return conflicts, False


def _find_phrase_conflicts_brute(
    embeddings, all_phrases, phrase_to_intent, phrase_conflict_max,
    phrase_chunk_size=256, progress_bar=None,
):
    """Brute-force pairwise comparison, chunked for memory efficiency."""
    total = len(all_phrases)
    phrase_chunk_size = max(int(phrase_chunk_size), 1)
    conflicts = []
    limit_reached = False
    processed = 0

    for start_idx in range(0, total, phrase_chunk_size):
        end_idx = min(start_idx + phrase_chunk_size, total)
        chunk_emb = embeddings[start_idx:end_idx]
        sim_block = cosine_similarity(chunk_emb, embeddings)

        for row_offset, phrase_index in enumerate(range(start_idx, end_idx)):
            phrase_intent = phrase_to_intent[phrase_index]
            row_sims = sim_block[row_offset]

            for other_index in range(phrase_index + 1, total):
                other_intent = phrase_to_intent[other_index]
                if other_intent == phrase_intent:
                    continue
                sim = row_sims[other_index]
                if sim >= PHRASE_SIMILARITY_MIN:
                    note = '⚠️ EXACT DUPLICATE' if all_phrases[phrase_index] == all_phrases[other_index] else ''
                    conflicts.append({
                        'Phrase': all_phrases[phrase_index],
                        'Intent': phrase_intent,
                        'Similar To': all_phrases[other_index],
                        'Other Intent': other_intent,
                        'Similarity': sim,
                        'Note': note,
                    })
                    if len(conflicts) >= phrase_conflict_max:
                        limit_reached = True
                        break
            if limit_reached:
                break

        processed += (end_idx - start_idx)
        if progress_bar:
            progress_bar.progress(min(processed / total, 0.99))
        if limit_reached:
            break

    if progress_bar:
        progress_bar.progress(1.0)
        progress_bar.empty()

    return conflicts, limit_reached


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
    pooling_strategy='mean',
    phrase_conflict_max=10_000,
    faiss_k=100,
):
    all_phrases = []
    phrase_to_intent = []
    for intent_name, phrases in intents.items():
        all_phrases.extend(phrases)
        phrase_to_intent.extend([intent_name] * len(phrases))

    embeddings = encode_texts(
        all_phrases, model, tokenizer, model_type,
        device_obj=device_obj,
        batch_size=embed_batch_size,
        normalize_embeddings=True,
        use_mixed_precision=use_mixed_precision,
        show_progress=show_progress,
        pooling_strategy=pooling_strategy,
    )

    # Intent-level embeddings (average of phrase embeddings per intent)
    intent_indices = {}
    for idx, intent_name in enumerate(phrase_to_intent):
        intent_indices.setdefault(intent_name, []).append(idx)

    intent_names = list(intent_indices.keys())
    intent_embeddings = {
        name: np.mean(embeddings[idx_list], axis=0)
        for name, idx_list in intent_indices.items()
    }
    intent_emb_matrix = np.array([intent_embeddings[n] for n in intent_names])
    intent_sim_matrix = cosine_similarity(intent_emb_matrix)

    # Phrase-level conflicts
    progress_bar = st.progress(0) if show_progress and len(all_phrases) > 0 else None

    if FAISS_AVAILABLE:
        phrase_conflicts, limit_reached = _find_phrase_conflicts_faiss(
            embeddings, all_phrases, phrase_to_intent,
            phrase_conflict_max, k=faiss_k,
        )
    else:
        phrase_conflicts, limit_reached = _find_phrase_conflicts_brute(
            embeddings, all_phrases, phrase_to_intent,
            phrase_conflict_max, phrase_chunk_size, progress_bar,
        )

    if progress_bar:
        progress_bar.progress(1.0)
        progress_bar.empty()

    return {
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
        'pooling_strategy': pooling_strategy,
        'faiss_used': FAISS_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Intent health scoring
# ---------------------------------------------------------------------------

def compute_intent_health(intents, embeddings, phrase_to_intent, intent_names):
    """Compute per-intent health scores combining cohesion, separation, size, and keyword diversity.

    Returns a list of dicts suitable for display in a dataframe.
    """
    from core.keywords import extract_keywords

    intent_indices = {}
    for idx, name in enumerate(phrase_to_intent):
        intent_indices.setdefault(name, []).append(idx)

    # Pre-compute intent centroid embeddings
    centroids = {}
    for name in intent_names:
        idxs = intent_indices[name]
        centroids[name] = np.mean(embeddings[idxs], axis=0, keepdims=True)

    records = []
    for name in intent_names:
        idxs = intent_indices[name]
        embs = embeddings[idxs]
        n_phrases = len(idxs)

        # Cohesion: average pairwise similarity within the intent (higher = tighter cluster)
        if n_phrases > 1:
            intra_sim = cosine_similarity(embs)
            # Upper triangle excluding diagonal
            mask = np.triu_indices_from(intra_sim, k=1)
            cohesion = float(np.mean(intra_sim[mask]))
        else:
            cohesion = 1.0

        # Separation: average similarity of this intent's centroid to other intent centroids
        # (lower = better separated)
        other_centroids = [centroids[n] for n in intent_names if n != name]
        if other_centroids:
            other_matrix = np.vstack(other_centroids)
            sep_sims = cosine_similarity(centroids[name], other_matrix)[0]
            separation = float(np.mean(sep_sims))
        else:
            separation = 0.0

        # Keyword diversity: unique keywords / total keyword tokens
        all_kw = []
        for phrase in intents[name]:
            all_kw.extend(extract_keywords(phrase))
        if all_kw:
            keyword_diversity = len(set(all_kw)) / len(all_kw)
        else:
            keyword_diversity = 0.0

        # Composite health score (0-100)
        #  - cohesion contributes positively (weight 35)
        #  - low separation contributes positively (weight 35)
        #  - phrase count contributes positively up to 50 (weight 15)
        #  - keyword diversity contributes positively (weight 15)
        size_score = min(n_phrases / 50.0, 1.0)
        health = (
            cohesion * 35
            + (1 - separation) * 35
            + size_score * 15
            + keyword_diversity * 15
        )

        if health >= 70:
            grade = '🟢 Good'
        elif health >= 50:
            grade = '🟡 Fair'
        else:
            grade = '🔴 Needs Work'

        records.append({
            'Intent': name,
            'Phrases': n_phrases,
            'Cohesion': round(cohesion, 3),
            'Separation': round(separation, 3),
            'Keyword Diversity': round(keyword_diversity, 3),
            'Health Score': round(health, 1),
            'Grade': grade,
        })

    records.sort(key=lambda r: r['Health Score'])
    return records


# ---------------------------------------------------------------------------
# Dimensionality reduction for visualisation
# ---------------------------------------------------------------------------

def compute_tsne(embeddings, perplexity=30, random_state=42):
    """Reduce embeddings to 2-D via t-SNE for scatter plot visualisation."""
    n_samples = embeddings.shape[0]
    perplexity = min(perplexity, max(5, n_samples - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_iter=1000)
    return tsne.fit_transform(embeddings)
