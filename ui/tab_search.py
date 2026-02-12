import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from collections import Counter

from core.memory import get_device
from core.search import semantic_search, batch_semantic_search
from core.keywords import extract_keywords
from utils.export import generate_search_excel, generate_search_json


def render_tab_search(intents, sidebar_cfg):
    st.header("Semantic Search & Dataset Audit")
    st.info("Search for semantically similar phrases to audit your dataset or determine where to add new phrases")

    if st.session_state.model is None:
        st.warning("Please load the model from the sidebar first")
        return
    if st.session_state.analysis_results is None:
        st.warning("Please run the similarity analysis first (in the Similarity Analysis tab)")
        return

    ar = st.session_state.analysis_results
    all_phrases = ar['all_phrases']
    phrase_to_intent = ar['phrase_to_intent']
    embeddings = ar['embeddings']
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    # ---- Search mode selector (new: single vs batch) ----
    search_mode = st.radio("Search Mode:", ["Single Phrase", "Batch Search (multiple phrases)"], horizontal=True)

    if search_mode == "Single Phrase":
        _render_single_search(all_phrases, phrase_to_intent, embeddings, intents, sidebar_cfg)
    else:
        _render_batch_search(all_phrases, phrase_to_intent, embeddings, intents, sidebar_cfg)

    # ---- Search history ----
    if st.session_state.search_history:
        st.markdown("---")
        with st.expander("Search History", expanded=False):
            history_df = pd.DataFrame(st.session_state.search_history).iloc[::-1]
            display_history = history_df[['timestamp', 'query', 'threshold', 'results', 'top_similarity', 'recommended_intent']].head(20)
            st.dataframe(display_history, width='stretch')

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export History", key="export_history"):
                    st.download_button("Download History CSV", history_df.to_csv(index=False), "search_history.csv", "text/csv", key="download_history")
            with col2:
                if st.button("Clear History", key="clear_search_history"):
                    st.session_state.search_history = []
                    st.rerun()


# ===========================================================================
# Single search
# ===========================================================================

def _render_single_search(all_phrases, phrase_to_intent, embeddings, intents, sidebar_cfg):
    col1, col2 = st.columns([2, 1])

    with col1:
        search_query = st.text_area(
            "Enter phrase to search or audit:",
            placeholder="e.g., 'I want to book a flight to Paris' or 'Cancel my subscription'",
            height=100,
            help="Enter any phrase to find semantically similar phrases across all intents.",
        )
        with st.expander("Example Use Cases", expanded=False):
            st.markdown("""
            **Dataset Auditing:** Search for existing phrases to find duplicates.
            **Adding New Phrases:** See which intent a new phrase best fits.
            **Quality Check:** Test edge cases to see how they match.
            """)

    with col2:
        st.markdown("### Search Settings")
        search_threshold = st.slider("Min similarity score:", 0.0, 1.0, 0.5, 0.05,
                                     help="Set to 0 to see all phrases ranked by similarity")
        top_k = st.number_input("Max results:", min_value=10, max_value=1000, value=50, step=10)
        include_exact = st.checkbox("Include exact matches", value=True)
        show_recommendations = st.checkbox("Show recommendations", value=True)

    # Persist results in session state
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = None
    if 'last_search_query' not in st.session_state:
        st.session_state.last_search_query = None
    if 'last_search_metadata' not in st.session_state:
        st.session_state.last_search_metadata = {}

    if st.button("Search & Analyze", type="primary", disabled=not search_query):
        with st.spinner("Searching..."):
            try:
                current_device = get_device(st.session_state.force_cpu)
                results = semantic_search(
                    search_query, embeddings, all_phrases, phrase_to_intent,
                    st.session_state.model, st.session_state.tokenizer,
                    st.session_state.model_type, current_device,
                    st.session_state.pooling_strategy,
                    threshold=search_threshold, top_k=top_k,
                )
                if not include_exact:
                    results = [r for r in results if r['phrase'].lower().strip() != search_query.lower().strip()]

                st.session_state.last_search_results = results
                st.session_state.last_search_query = search_query
                st.session_state.last_search_metadata = {
                    'threshold': search_threshold, 'top_k': top_k,
                    'include_exact': include_exact, 'show_recommendations': show_recommendations,
                }
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
                st.session_state.last_search_results = None

    search_results = st.session_state.last_search_results
    if search_results is None:
        return

    search_query = st.session_state.last_search_query
    metadata = st.session_state.last_search_metadata
    search_threshold = metadata.get('threshold', 0.5)
    show_recommendations = metadata.get('show_recommendations', True)

    if not search_results:
        st.warning(f"No phrases found with similarity >= {search_threshold:.2f}")
        st.info("Try lowering the similarity threshold")
        return

    st.success(f"Found {len(search_results)} similar phrases with similarity >= {search_threshold:.2f}")

    # ---- Recommendations ----
    dominant_intent = None
    confidence = None
    if show_recommendations:
        st.markdown("---")
        st.subheader("Recommendations")

        top_match = search_results[0]
        top_similarity = top_match['similarity']
        top_intent = top_match['intent']

        top_10_intents = [r['intent'] for r in search_results[:min(10, len(search_results))]]
        intent_freq = Counter(top_10_intents)
        dominant_intent = intent_freq.most_common(1)[0][0]
        dominant_count = intent_freq[dominant_intent]
        confidence = dominant_count / min(10, len(search_results))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Match Similarity", f"{top_similarity:.3f}")
            if top_similarity > 0.95:
                st.caption("Very High -- Likely duplicate")
            elif top_similarity > 0.85:
                st.caption("High -- Good fit")
            elif top_similarity > 0.70:
                st.caption("Moderate -- Acceptable fit")
            else:
                st.caption("Low -- Consider new intent")
        with col2:
            st.metric("Best Matching Intent", dominant_intent)
            st.caption(f"{dominant_count}/10 top matches")
        with col3:
            st.metric("Confidence", f"{confidence:.0%}")
            st.caption("High confidence" if confidence > 0.7 else "Mixed results")

        st.markdown("#### Action Recommendation:")
        if top_similarity > 0.98:
            st.error(f"**DUPLICATE DETECTED:** Nearly identical to phrases in **{top_intent}**. Do not add.")
            for r in search_results[:3]:
                if r['similarity'] > 0.95:
                    st.markdown(f"- *\"{r['phrase']}\"* (similarity: {r['similarity']:.3f})")
        elif top_similarity > 0.85 and confidence > 0.7:
            st.success(f"**ADD TO INTENT:** Add this phrase to **{dominant_intent}**")
            st.markdown(f"High similarity ({top_similarity:.3f}) and strong consensus ({confidence:.0%}).")
        elif top_similarity > 0.70:
            st.warning(f"**REVIEW NEEDED:** Consider adding to **{dominant_intent}**, but review carefully")
            if len(intent_freq) > 1:
                st.markdown("Similar phrases found across multiple intents:")
                for intent, count in intent_freq.most_common(3):
                    st.markdown(f"- {intent}: {count} matches")
        else:
            st.info(f"**NEW INTENT SUGGESTED:** Low similarity ({top_similarity:.3f}) to existing phrases")

    st.markdown("---")

    # ---- Build results table ----
    results_data = []
    for result in search_results:
        sim = result['similarity']
        phrase = result['phrase']
        if phrase.lower().strip() == search_query.lower().strip():
            match_type = "Exact"
        elif sim > 0.95:
            match_type = "Very High"
        elif sim > 0.85:
            match_type = "High"
        elif sim > 0.75:
            match_type = "Moderate"
        elif sim > 0.60:
            match_type = "Low"
        else:
            match_type = "Very Low"
        results_data.append({
            'Similarity Score': sim,
            'Match Type': match_type,
            'Intent': result['intent'],
            'Phrase': phrase,
            'Keywords': ', '.join(extract_keywords(phrase)[:5]),
        })

    results_df = pd.DataFrame(results_data)
    results_df['Similarity Score Display'] = results_df['Similarity Score'].apply(lambda x: f"{x:.4f}")

    # Statistics
    st.subheader("Search Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Unique Intents", results_df['Intent'].nunique())
    with col2:
        st.metric("Very Similar (>0.95)", len(results_df[results_df['Similarity Score'] > 0.95]))
    with col3:
        st.metric("High Similar (0.85-0.95)", len(results_df[(results_df['Similarity Score'] > 0.85) & (results_df['Similarity Score'] <= 0.95)]))
    with col4:
        st.metric("Avg Similarity", f"{results_df['Similarity Score'].mean():.3f}")

    # Intent distribution
    st.subheader("Results by Intent")
    intent_counts = results_df['Intent'].value_counts()

    with st.expander("Intent Distribution Analysis", expanded=True):
        intent_dist = []
        for intent, count in intent_counts.items():
            ir = results_df[results_df['Intent'] == intent]
            intent_dist.append({
                'Intent': intent, 'Count': count,
                'Avg Similarity': f"{ir['Similarity Score'].mean():.3f}",
                'Max Similarity': f"{ir['Similarity Score'].max():.3f}",
                'Min Similarity': f"{ir['Similarity Score'].min():.3f}",
                '% of Results': f"{count / len(results_df) * 100:.1f}%",
            })
        intent_dist_df = pd.DataFrame(intent_dist)
        st.dataframe(intent_dist_df, width='stretch')

    # Results table
    st.subheader("All Similar Phrases (Ranked by Similarity)")
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_intent = st.selectbox("Filter by intent:", ["All"] + list(intent_counts.index), key="search_filter_intent")
    with col2:
        filter_match = st.multiselect(
            "Filter by match type:",
            ["Exact", "Very High", "High", "Moderate", "Low", "Very Low"],
            default=["Exact", "Very High", "High", "Moderate", "Low", "Very Low"],
            key="search_filter_match",
        )
    with col3:
        display_limit = st.selectbox("Display rows:", [25, 50, 100, 200, 500], index=1, key="search_display_limit")

    filtered = results_df.copy()
    if filter_intent != "All":
        filtered = filtered[filtered['Intent'] == filter_intent]
    if filter_match:
        filtered = filtered[filtered['Match Type'].isin(filter_match)]

    display = filtered[['Similarity Score Display', 'Match Type', 'Intent', 'Phrase', 'Keywords']].head(display_limit)
    display = display.rename(columns={'Similarity Score Display': 'Similarity Score'})
    if not display.empty:
        st.dataframe(display, width='stretch', height=400)
        if len(filtered) > display_limit:
            st.info(f"Showing {display_limit} of {len(filtered)} filtered results")
    else:
        st.warning("No results match the current filters")

    # ---- Export ----
    st.subheader("Export Search Results")
    col1, col2, col3 = st.columns(3)
    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    with col1:
        st.download_button("Download as CSV", results_df.to_csv(index=False),
                           f"semantic_search_results_{ts}.csv", "text/csv", key="download_search_csv")
    with col2:
        excel_out = generate_search_excel(search_query, search_threshold, results_df, intent_dist_df, confidence, search_results)
        st.download_button("Download Excel Report", excel_out,
                           f"semantic_search_audit_{ts}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="download_search_excel")
    with col3:
        json_data = generate_search_json(search_query, search_threshold, results_df, search_results, dominant_intent)
        st.download_button("Download as JSON", json_data,
                           f"semantic_search_results_{ts}.json", "application/json", key="download_search_json")

    # ---- Visualisations ----
    st.subheader("Visualizations")
    fig = px.histogram(results_df, x='Similarity Score', nbins=20, title='Distribution of Similarity Scores',
                       labels={'count': 'Number of Phrases', 'Similarity Score': 'Similarity Score'},
                       color_discrete_sequence=['#636EFA'])
    fig.add_vline(x=0.85, line_dash="dash", line_color="red", annotation_text="High Threshold")
    fig.add_vline(x=0.70, line_dash="dash", line_color="yellow", annotation_text="Moderate Threshold")
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

    if len(intent_counts) > 1:
        fig2 = px.pie(values=intent_counts.values, names=intent_counts.index, title='Results Distribution by Intent', height=400)
        st.plotly_chart(fig2, width='stretch')

    # Save to history
    if st.button("Save to Search History", key="save_search"):
        st.session_state.search_history.append({
            'query': search_query, 'threshold': search_threshold,
            'results': len(search_results),
            'top_similarity': search_results[0]['similarity'] if search_results else 0,
            'recommended_intent': dominant_intent or "N/A",
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        st.success("Saved to search history")


# ===========================================================================
# Batch search (new feature)
# ===========================================================================

def _render_batch_search(all_phrases, phrase_to_intent, embeddings, intents, sidebar_cfg):
    st.markdown("### Batch Phrase Audit")
    st.info("Paste multiple phrases (one per line) to audit them all at once against your dataset.")

    col1, col2 = st.columns([3, 1])
    with col1:
        batch_input = st.text_area(
            "Enter phrases (one per line):",
            placeholder="I want to book a flight\nCancel my subscription\nWhat is the weather today\n...",
            height=200,
            key="batch_search_input",
        )
    with col2:
        batch_threshold = st.slider("Min similarity:", 0.0, 1.0, 0.5, 0.05, key="batch_threshold")
        batch_top_k = st.number_input("Top-K per query:", min_value=5, max_value=100, value=10, step=5, key="batch_top_k")

    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None

    if st.button("Run Batch Audit", type="primary", disabled=not batch_input.strip()):
        queries = [q.strip() for q in batch_input.strip().split('\n') if q.strip()]
        if not queries:
            st.warning("No phrases entered.")
            return

        with st.spinner(f"Auditing {len(queries)} phrases..."):
            current_device = get_device(st.session_state.force_cpu)
            results = batch_semantic_search(
                queries, embeddings, all_phrases, phrase_to_intent,
                st.session_state.model, st.session_state.tokenizer,
                st.session_state.model_type, current_device,
                st.session_state.pooling_strategy,
                threshold=batch_threshold, top_k=batch_top_k,
            )
            st.session_state.batch_results = results

    results = st.session_state.batch_results
    if results is None:
        return

    st.success(f"Audited {len(results)} phrases")

    # Summary table
    summary_rows = []
    for r in results:
        action_map = {
            'DUPLICATE': 'DUPLICATE -- Do not add',
            'ADD': f"ADD to {r['recommended_intent']}",
            'REVIEW': f"REVIEW -- possibly {r['recommended_intent']}",
            'NEW_INTENT': 'NEW INTENT suggested',
        }
        summary_rows.append({
            'Phrase': r['query'][:80],
            'Top Similarity': f"{r['top_similarity']:.3f}",
            'Recommended Intent': r['recommended_intent'] or 'N/A',
            'Action': action_map.get(r['action'], r['action']),
            'Matches Found': len(r['matches']),
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, width='stretch', height=min(600, 35 * len(summary_rows) + 38))

    # Action breakdown
    action_counts = Counter(r['action'] for r in results)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duplicates", action_counts.get('DUPLICATE', 0))
    with col2:
        st.metric("Ready to Add", action_counts.get('ADD', 0))
    with col3:
        st.metric("Need Review", action_counts.get('REVIEW', 0))
    with col4:
        st.metric("New Intents", action_counts.get('NEW_INTENT', 0))

    # Download
    st.download_button(
        "Download Batch Audit Report",
        summary_df.to_csv(index=False),
        f"batch_audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv", key="download_batch_audit",
    )

    # Expandable per-query details
    with st.expander("Detailed per-phrase matches", expanded=False):
        for r in results:
            st.markdown(f"**{r['query'][:80]}** -- Action: `{r['action']}` -- Top: {r['top_similarity']:.3f}")
            if r['matches']:
                match_df = pd.DataFrame(r['matches'][:5])[['similarity', 'intent', 'phrase']]
                st.dataframe(match_df, width='stretch')
            else:
                st.caption("No matches above threshold")
