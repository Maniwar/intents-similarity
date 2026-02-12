import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from core.embeddings import encode_texts
import logging

logger = logging.getLogger(__name__)


def semantic_search(
    query, embeddings, all_phrases, phrase_to_intent,
    model, tokenizer, model_type, device_obj, pooling_strategy,
    threshold=0.7, top_k=100,
):
    query_embedding = encode_texts(
        [query], model, tokenizer, model_type,
        device_obj=device_obj, batch_size=1,
        normalize_embeddings=True,
        use_mixed_precision=device_obj.type == 'cuda',
        pooling_strategy=pooling_strategy,
    )

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    results = []
    for idx, (phrase, intent, sim) in enumerate(zip(all_phrases, phrase_to_intent, similarities)):
        if sim >= threshold:
            results.append({
                'similarity': float(sim),
                'phrase': phrase,
                'intent': intent,
                'idx': idx,
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


def batch_semantic_search(
    queries, embeddings, all_phrases, phrase_to_intent,
    model, tokenizer, model_type, device_obj, pooling_strategy,
    threshold=0.7, top_k=20, batch_size=32,
):
    """Run semantic search for multiple queries at once.

    Returns a list of dicts -- one per query -- each containing the query text,
    the recommended intent, the top similarity score, and a list of result dicts.
    """
    query_embeddings = encode_texts(
        queries, model, tokenizer, model_type,
        device_obj=device_obj, batch_size=batch_size,
        normalize_embeddings=True,
        use_mixed_precision=device_obj.type == 'cuda',
        pooling_strategy=pooling_strategy,
    )

    sim_matrix = cosine_similarity(query_embeddings, embeddings)  # (n_queries, n_phrases)

    batch_results = []
    for q_idx, query in enumerate(queries):
        sims = sim_matrix[q_idx]
        matches = []
        for p_idx in range(len(all_phrases)):
            if sims[p_idx] >= threshold:
                matches.append({
                    'similarity': float(sims[p_idx]),
                    'phrase': all_phrases[p_idx],
                    'intent': phrase_to_intent[p_idx],
                    'idx': p_idx,
                })
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        matches = matches[:top_k]

        # Determine recommended intent from top matches
        if matches:
            from collections import Counter
            intent_counts = Counter(m['intent'] for m in matches[:10])
            recommended_intent = intent_counts.most_common(1)[0][0]
            top_sim = matches[0]['similarity']
        else:
            recommended_intent = None
            top_sim = 0.0

        # Classify action
        if top_sim > 0.98:
            action = 'DUPLICATE'
        elif top_sim > 0.85:
            action = 'ADD'
        elif top_sim > 0.70:
            action = 'REVIEW'
        else:
            action = 'NEW_INTENT'

        batch_results.append({
            'query': query,
            'recommended_intent': recommended_intent,
            'top_similarity': top_sim,
            'action': action,
            'matches': matches,
        })

    return batch_results
