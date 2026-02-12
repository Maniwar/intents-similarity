import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch

from core.memory import get_device, get_dynamic_limits
from core.analysis import perform_analysis, compute_intent_health, compute_tsne, _TSNE_SAMPLE_CAP, PHRASE_SIMILARITY_MIN
from core.keywords import analyze_keyword_overlap, extract_keywords_tfidf
from core.recommendations import build_priority_actions, build_phrase_issue_actions, build_phrase_recommendations
from core.paraphrase import PARAPHRASE_MODELS, generate_paraphrases
from utils.data_loader import (
    find_cross_intent_duplicates,
    remove_cross_intent_duplicates,
    validate_intents,
)
from utils.export import generate_excel_workbook, generate_text_report
from ui.components import render_metric_row, render_data_quality_warnings, filter_phrase_conflicts


def render_tab_similarity(df, intents, sidebar_cfg):
    similarity_threshold = sidebar_cfg['similarity_threshold']
    phrase_similarity_threshold = sidebar_cfg['phrase_similarity_threshold']
    dynamic_limits = sidebar_cfg['dynamic_limits']

    st.header("Intent Data")
    st.dataframe(df, width='stretch')

    # ---- Data verification ----
    if intents:
        with st.expander("Verify Data Structure", expanded=False):
            st.write("**First 3 phrases from each intent:**")
            for intent_name, phrases_list in list(intents.items())[:5]:
                st.write(f"**{intent_name}:** ({len(phrases_list)} total phrases)")
                for idx, phrase in enumerate(phrases_list[:3], 1):
                    display = f"  {idx}. {phrase[:100]}..." if len(phrase) > 100 else f"  {idx}. {phrase}"
                    st.write(display)
                if len(phrases_list) > 3:
                    st.write(f"  ... and {len(phrases_list) - 3} more")

    # ---- Data validation (new) ----
    validation_warnings = validate_intents(intents)
    if validation_warnings:
        with st.expander(f"Data Quality Report ({len(validation_warnings)} issue(s))", expanded=True):
            render_data_quality_warnings(validation_warnings)

    # ---- Cross-intent duplicate detection (fixed) ----
    cross_dupes = find_cross_intent_duplicates(intents)
    if cross_dupes:
        st.warning(f"Found {len(cross_dupes)} phrase(s) appearing in multiple intents. This may inflate similarity scores.")
        with st.expander(f"View {len(cross_dupes)} Cross-Intent Duplicate Phrases", expanded=False):
            st.dataframe(pd.DataFrame(cross_dupes), width='stretch')

            if st.button("Remove Cross-Intent Duplicates (keep first occurrence)"):
                cleaned_df = remove_cross_intent_duplicates(df, intents)
                st.session_state.data = cleaned_df
                st.session_state.analysis_results = None
                st.session_state.embeddings = None
                st.success("Removed cross-intent duplicates! Re-run analysis to see improvements.")
                st.download_button(
                    "Download Cleaned Dataset", cleaned_df.to_csv(index=False),
                    "cleaned_intents.csv", "text/csv",
                )

    # ---- Large dataset warning ----
    total_phrases = sum(len(p) for p in intents.values())
    if total_phrases > 1000:
        st.warning(f"Large dataset detected ({total_phrases} phrases). Analysis will use smart indexing to manage resources.")
        st.info("**Tip**: For large datasets, increase the similarity threshold to focus on the most critical issues first.")

    # ---- Analysis ----
    if st.session_state.model is None:
        st.warning("Please load the model from the sidebar first")
        return

    st.header("Similarity Analysis")

    if st.button("Run Analysis", key="run_analysis_tab1"):
        current_device = get_device(st.session_state.force_cpu)
        with st.spinner("Computing embeddings and similarities..."):
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
                pooling_strategy=st.session_state.pooling_strategy,
                phrase_conflict_max=dynamic_limits['phrase_conflict_max'],
            )
            st.session_state.analysis_results = analysis_results
            st.session_state.embeddings = analysis_results['embeddings']

            backend = "FAISS index" if analysis_results.get('faiss_used') else "brute-force"
            st.success(f"Analysis complete! (phrase search: {backend})")

    if not st.session_state.analysis_results:
        st.info("Click **Run Analysis** to compute similarities")
        return

    # ---- Unpack results ----
    ar = st.session_state.analysis_results
    all_phrases = ar['all_phrases']
    phrase_to_intent = ar['phrase_to_intent']
    embeddings = ar['embeddings']
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    intent_names = ar['intent_names']
    intent_sim_matrix = ar['intent_sim_matrix']
    if isinstance(intent_sim_matrix, list):
        intent_sim_matrix = np.array(intent_sim_matrix)

    # ---- Confusing pairs ----
    confusing_pairs = []
    for i in range(len(intent_names)):
        for j in range(i + 1, len(intent_names)):
            sim = intent_sim_matrix[i][j]
            if sim >= similarity_threshold:
                confusing_pairs.append({
                    'Intent A': intent_names[i],
                    'Intent B': intent_names[j],
                    'Similarity': f"{sim:.3f}",
                    'Risk': 'High' if sim > 0.9 else 'Medium',
                })

    # ---- Executive summary ----
    st.subheader("Executive Summary")
    render_metric_row([
        {'label': 'Total Intents', 'value': len(intent_names)},
        {'label': 'Total Phrases', 'value': len(all_phrases)},
        {'label': 'Unique Phrases', 'value': len(set(all_phrases))},
        {'label': 'Confusing Pairs', 'value': len(confusing_pairs)},
    ])

    # ---- Intent health scoring (new) ----
    st.subheader("Intent Health Dashboard")
    health_records = compute_intent_health(intents, embeddings, phrase_to_intent, intent_names)
    health_df = pd.DataFrame(health_records)
    st.dataframe(health_df, width='stretch', height=min(400, 35 * len(health_records) + 38))
    st.caption("Health Score (0-100) combines cohesion (intra-intent similarity), separation (inter-intent distance), phrase count, and keyword diversity.")

    # ---- Heatmap ----
    st.subheader("Intent Similarity Matrix")
    fig = px.imshow(
        intent_sim_matrix, x=intent_names, y=intent_names,
        color_continuous_scale='RdYlGn_r',
        labels=dict(color="Similarity"),
        title="Intent Similarity Matrix",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width='stretch')

    # ---- t-SNE visualisation (new) ----
    st.subheader("Phrase Embedding Space (t-SNE)")
    n_phrases = embeddings.shape[0]
    if n_phrases > _TSNE_SAMPLE_CAP:
        st.info(f"Large dataset ({n_phrases:,} phrases) — t-SNE computed on a {_TSNE_SAMPLE_CAP:,}-point sample; remaining points interpolated.")
    with st.spinner("Computing 2-D projection..."):
        coords = compute_tsne(embeddings)
    tsne_df = pd.DataFrame({
        'x': coords[:, 0], 'y': coords[:, 1],
        'Intent': phrase_to_intent,
        'Phrase': [p[:80] for p in all_phrases],
    })
    fig_tsne = px.scatter(
        tsne_df, x='x', y='y', color='Intent',
        hover_data=['Phrase'],
        title='Phrase Clusters (overlapping clouds = confusing intents)',
        height=650,
    )
    fig_tsne.update_traces(marker=dict(size=5, opacity=0.7))
    fig_tsne.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.3))
    st.plotly_chart(fig_tsne, width='stretch')

    # ---- Confusing pairs table ----
    if confusing_pairs:
        st.subheader("Potentially Confusing Intent Pairs")
        st.dataframe(pd.DataFrame(confusing_pairs), width='stretch')
    else:
        st.success("No confusing intent pairs detected at this threshold!")

    # ---- Phrase-level analysis ----
    st.subheader("Cross-Intent Phrase Confusion")
    phrase_conflicts_all = ar['phrase_conflicts']
    limit_reached = ar['phrase_conflict_limit_reached']

    col1, col2, col3 = st.columns(3)
    with col1:
        min_sim_filter = st.slider(
            "Show only phrases above similarity:",
            min_value=PHRASE_SIMILARITY_MIN, max_value=1.0,
            value=max(phrase_similarity_threshold, PHRASE_SIMILARITY_MIN), step=0.05,
            key="tab1_phrase_filter",
        )
    with col2:
        max_display_rows = st.selectbox("Max rows to display:", [50, 100, 500, 1000], index=1, key="tab1_max_rows")
    with col3:
        specific_intent_filter = st.selectbox(
            "Filter by intent:", ["All Intents"] + list(intents.keys()), key="tab1_intent_filter",
        )

    if limit_reached:
        st.warning(f"Phrase conflict analysis capped at {dynamic_limits['phrase_conflict_max']:,} results "
                    f"(system memory: {dynamic_limits['available_gb']}GB -- {dynamic_limits['memory_tier']} tier).")

    phrase_confusion = filter_phrase_conflicts(phrase_conflicts_all, min_sim_filter, specific_intent_filter)

    if phrase_confusion:
        st.warning(f"Found {len(phrase_confusion)} phrase conflicts above {min_sim_filter:.2f} similarity")
        confusion_df = pd.DataFrame(phrase_confusion)

        # Separate exact duplicates
        if 'Note' in confusion_df.columns:
            exact_dupes = confusion_df[confusion_df['Note'] == '⚠️ EXACT DUPLICATE']
            similar_phrases = confusion_df[confusion_df['Note'] != '⚠️ EXACT DUPLICATE']

            if len(exact_dupes) > 0:
                st.error(f"CRITICAL DATA ISSUE: Found {len(exact_dupes)} exact duplicate phrases across different intents!")
                with st.expander(f"View ALL {len(exact_dupes)} Exact Duplicates", expanded=False):
                    st.dataframe(exact_dupes, width='stretch', height=400)
                    st.download_button(
                        f"Download ALL Exact Duplicates ({len(exact_dupes)} rows)",
                        exact_dupes.to_csv(index=False), "exact_duplicates.csv", "text/csv",
                        key='download-exact-dupes-tab1',
                    )

            st.info(f"Found {len(similar_phrases)} similar (but not identical) phrase pairs")
            confusion_df = similar_phrases

        confusion_df['Similarity_Float'] = confusion_df['Similarity'].astype(float)
        confusion_df = confusion_df.sort_values('Similarity_Float', ascending=False).drop('Similarity_Float', axis=1)

        st.dataframe(confusion_df.head(max_display_rows), width='stretch')
        if len(confusion_df) > max_display_rows:
            st.warning(f"{len(confusion_df) - max_display_rows} additional conflicts not shown.")
        st.download_button(
            f"Download Full Confusion Report ({len(confusion_df)} rows)",
            confusion_df.to_csv(index=False), "phrase_confusion_report.csv", "text/csv",
            key='download-confusion-tab1',
        )
    else:
        st.success(f"No phrase confusion detected above {min_sim_filter:.2f} similarity!")

    # ======================================================================
    # RECOMMENDATIONS
    # ======================================================================
    st.header("Actionable Recommendations")

    # Quick action summary
    st.markdown("### Quick Action Summary")
    critical_pairs = len([p for p in confusing_pairs if float(p['Similarity']) > 0.92])
    high_sim_pairs = len([p for p in confusing_pairs if 0.85 <= float(p['Similarity']) <= 0.92])
    phrase_critical = len([p for p in phrase_confusion[:1000] if float(p['Similarity']) > 0.95]) if phrase_confusion else 0

    total_conflicts = len(phrase_confusion) if phrase_confusion else 0
    conflict_display = f"{dynamic_limits['phrase_conflict_max']:,}+" if (limit_reached and total_conflicts >= dynamic_limits['phrase_conflict_max']) else f"{total_conflicts:,}"

    render_metric_row([
        {'label': 'Must Fix', 'value': critical_pairs},
        {'label': 'Should Fix', 'value': high_sim_pairs},
        {'label': 'Phrases to Move/Delete', 'value': f"{phrase_critical}+"},
        {'label': 'Total Conflicts', 'value': conflict_display},
    ])
    st.markdown("---")

    # Baseline
    if st.session_state.baseline_results is None:
        if st.button("Set as Baseline (for before/after comparison)", key="set_baseline_tab1"):
            st.session_state.baseline_results = {
                'confusing_pairs': confusing_pairs,
                'phrase_confusion': phrase_confusion,
                'intent_sim_matrix': intent_sim_matrix,
            }
            st.success("Baseline saved! Make changes and re-run to see improvements.")
    else:
        st.info("Baseline comparison available -- see below")

    # Priority actions
    st.subheader("Priority Actions (Ranked by Impact)")
    priority_actions = build_priority_actions(confusing_pairs, intents)
    priority_actions += build_phrase_issue_actions(confusing_pairs, phrase_confusion, intents)

    if priority_actions:
        priority_df = pd.DataFrame(priority_actions)
        st.dataframe(priority_df, width='stretch', height=400)
        st.download_button(
            f"Download All Priority Actions ({len(priority_df)} items)",
            priority_df.to_csv(index=False), "priority_actions.csv", "text/csv",
            key='download-priority-tab1',
        )
    else:
        st.success("No priority actions needed!")

    st.markdown("---")

    # Phrase-level recommendations
    st.subheader("Phrase-Level Actions")
    if phrase_confusion:
        # Sort by similarity descending so the most critical conflicts are
        # analysed first (build_phrase_recommendations caps at 500).
        phrase_confusion_sorted = sorted(phrase_confusion, key=lambda c: float(c['Similarity']), reverse=True)
        st.info(f"Analysing top {min(len(phrase_confusion_sorted), 500)} phrase conflicts for actionable recommendations...")
        phrase_recs = build_phrase_recommendations(phrase_confusion_sorted)

        phrase_columns = [
            'Action', 'Your Phrase', 'Current Intent', 'Conflicts With Phrase',
            'Other Intent', 'Similarity', 'Reason', 'Common Words', 'Add to Current', 'Suggestion',
        ]
        phrase_df = pd.DataFrame(phrase_recs, columns=phrase_columns)
        if not phrase_df.empty:
            phrase_df['Sim_Sort'] = phrase_df['Similarity'].astype(float)
            phrase_df = phrase_df.sort_values('Sim_Sort', ascending=False).drop('Sim_Sort', axis=1)

        col1, col2 = st.columns(2)
        with col1:
            action_filter = st.multiselect(
                "Filter by action type:",
                ["🔴 MOVE or DELETE", "🟡 REPHRASE", "🟢 REVIEW"],
                default=["🔴 MOVE or DELETE", "🟡 REPHRASE"],
                key="action_filter_tab1",
            )
        with col2:
            display_limit = st.selectbox("Rows to display:", [20, 50, 250, 2000], index=1, key="phrase_display_limit_tab1")

        filtered_df = phrase_df[phrase_df['Action'].isin(action_filter)] if action_filter else phrase_df
        display_df = filtered_df.head(display_limit)

        if display_df.empty:
            st.warning("No phrase-level actions match the current filters.")
        else:
            st.dataframe(display_df, width='stretch', height=400)
            st.info("**How to read this table:** 'Your Phrase' is the problem phrase. 'Conflicts With Phrase' is the similar phrase in the other intent.")
            if len(filtered_df) > display_limit:
                st.info(f"Showing {display_limit} of {len(filtered_df)} recommendations. Download full list below.")

        st.download_button(
            f"Download All Phrase-Level Actions ({len(phrase_df)} rows)",
            phrase_df.to_csv(index=False), "phrase_level_actions.csv", "text/csv",
            key='download-phrase-actions-tab1',
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Critical Actions", len(phrase_df[phrase_df['Action'] == '🔴 MOVE or DELETE']))
        with col2:
            st.metric("Rephrase Needed", len(phrase_df[phrase_df['Action'] == '🟡 REPHRASE']))
        with col3:
            st.metric("Review", len(phrase_df[phrase_df['Action'] == '🟢 REVIEW']))
    else:
        st.success("No phrase-level conflicts detected!")

    st.markdown("---")

    # ---- TF-IDF keyword analysis (new) ----
    st.subheader("TF-IDF Keyword Analysis")
    tfidf_results = extract_keywords_tfidf(intents)
    if confusing_pairs:
        st.write("**Top discriminative terms per intent (TF-IDF bigrams):**")
        for pair in confusing_pairs[:5]:
            ia, ib = pair['Intent A'], pair['Intent B']
            from core.keywords import analyze_keyword_overlap
            kw = analyze_keyword_overlap(intents[ia], intents[ib])

            with st.expander(f"{ia} <-> {ib} (Similarity: {pair['Similarity']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Common Keywords ({len(kw['common_keywords'])}):**")
                    for w in kw['common_keywords'][:10]:
                        st.markdown(f"- `{w}`")
                with col2:
                    st.markdown(f"**Unique to '{ia}':**")
                    tfidf_ia = tfidf_results.get(ia, [])[:10]
                    for term, score in tfidf_ia:
                        st.markdown(f"- `{term}` ({score:.2f})")
                with col3:
                    st.markdown(f"**Unique to '{ib}':**")
                    tfidf_ib = tfidf_results.get(ib, [])[:10]
                    for term, score in tfidf_ib:
                        st.markdown(f"- `{term}` ({score:.2f})")

    st.markdown("---")

    # Merge vs differentiate
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Should You Merge?")
        merge_candidates = [p for p in confusing_pairs if float(p['Similarity']) > 0.92]
        if merge_candidates:
            st.warning(f"Found {len(merge_candidates)} intent pair(s) that might be duplicates:")
            for pair in merge_candidates[:3]:
                st.markdown(f"- **{pair['Intent A']}** <-> **{pair['Intent B']}** ({pair['Similarity']})")
            st.info("**Consider merging** if these intents serve the same business purpose")
        else:
            st.success("No clear merge candidates!")

    with col2:
        st.subheader("How to Differentiate")
        diff_candidates = [p for p in confusing_pairs if 0.85 <= float(p['Similarity']) <= 0.92]
        if diff_candidates:
            st.info(f"Found {len(diff_candidates)} intent pair(s) that need clearer separation:")
            for pair in diff_candidates[:3]:
                st.markdown(f"- **{pair['Intent A']}** <-> **{pair['Intent B']}** ({pair['Similarity']})")
            st.info("**Add distinctive phrases** using unique keywords")
        else:
            st.success("All distinct intents are well-separated!")

    st.markdown("---")

    # Before/after
    if st.session_state.baseline_results is not None:
        st.subheader("Before/After Comparison")
        baseline = st.session_state.baseline_results

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confusing Pairs", len(confusing_pairs),
                       delta=len(confusing_pairs) - len(baseline['confusing_pairs']),
                       delta_color="inverse")
        with col2:
            st.metric("Phrase Conflicts", len(phrase_confusion),
                       delta=len(phrase_confusion) - len(baseline['phrase_confusion']),
                       delta_color="inverse")
        with col3:
            cur_avg = np.mean([float(p['Similarity']) for p in confusing_pairs]) if confusing_pairs else 0
            base_avg = np.mean([float(p['Similarity']) for p in baseline['confusing_pairs']]) if baseline['confusing_pairs'] else 0
            st.metric("Avg Confusion Score", f"{cur_avg:.3f}", delta=f"{cur_avg - base_avg:.3f}", delta_color="inverse")

        if len(confusing_pairs) < len(baseline['confusing_pairs']):
            st.success("Great progress! You've reduced confusion between intents.")
        elif len(confusing_pairs) > len(baseline['confusing_pairs']):
            st.warning("Confusion increased. Review recent changes.")
        else:
            st.info("No change from baseline.")

        if st.button("Reset Baseline", key="reset_baseline_tab1"):
            st.session_state.baseline_results = None
            st.rerun()

    # ---- Phrase generation ----
    st.header("Generate Improved Training Phrases")
    st.info("Generate new phrases to increase diversity and reduce confusion")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_intent = st.selectbox("Select Intent:", list(intents.keys()), key="gen_intent_tab1")
    with col2:
        generation_strategy = st.selectbox(
            "Generation Strategy:", ["Diverse Variations", "Targeted Differentiation"],
            help="Diverse: general variations. Targeted: emphasize unique keywords",
            key="gen_strategy_tab1",
        )
    with col3:
        selected_paraphrase_model = st.selectbox(
            "Paraphrase Model:", list(PARAPHRASE_MODELS.keys()),
            format_func=lambda x: PARAPHRASE_MODELS[x],
            key="paraphrase_model_tab1",
        )

    num_generations = st.slider("Number of variations per phrase:", 1, 5, 2, key="num_gen_tab1")

    if generation_strategy == "Targeted Differentiation":
        confused_with = []
        for pair in confusing_pairs:
            if pair['Intent A'] == selected_intent:
                confused_with.append(pair['Intent B'])
            elif pair['Intent B'] == selected_intent:
                confused_with.append(pair['Intent A'])
        if confused_with:
            st.warning(f"'{selected_intent}' is confused with: {', '.join(confused_with)}")
            from core.keywords import analyze_keyword_overlap as _akw
            all_unique = set()
            for ci in confused_with:
                kw = _akw(intents[selected_intent], intents[ci])
                all_unique.update(kw['intent1_unique'][:5])
            if all_unique:
                st.info(f"Try to include these unique keywords: **{', '.join(list(all_unique)[:10])}**")
        else:
            st.success(f"'{selected_intent}' has no confusion -- generating diverse variations")

    if st.button("Generate Phrases", key="generate_phrases_tab1"):
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            st.error("Missing required dependency: `sentencepiece`")
            st.markdown("```bash\npip install sentencepiece protobuf\n```\nThen restart.")
            st.stop()

        use_gpu = torch.cuda.is_available() and not st.session_state.force_cpu
        with st.spinner(f"Loading {PARAPHRASE_MODELS[selected_paraphrase_model]} and generating phrases..."):
            try:
                new_phrases, was_limited = generate_paraphrases(
                    intents[selected_intent],
                    selected_paraphrase_model,
                    use_gpu=use_gpu,
                    num_variations=num_generations,
                )
                if was_limited:
                    st.warning(f"Generated variations for first 20 phrases only (out of {len(intents[selected_intent])}).")

                st.session_state.generated_phrases[selected_intent] = {
                    'phrases': new_phrases,
                    'strategy': generation_strategy,
                    'num_variations': num_generations,
                    'generated_at': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model_used': selected_paraphrase_model,
                }
                st.success(f"Generated {len(new_phrases)} new phrases!")
            except Exception as e:
                st.error(f"Error generating phrases: {str(e)}")
                with st.expander("Troubleshooting Tips", expanded=True):
                    st.markdown("""
                    **Common Issues:**
                    1. **Missing SentencePiece**: `pip install sentencepiece protobuf`
                    2. **First run**: model downloads ~500MB-3GB
                    3. **Memory**: reduce variations or use CPU
                    4. **Upgrade**: `pip install --upgrade transformers sentencepiece protobuf torch`
                    """)
                st.session_state.generated_phrases.pop(selected_intent, None)

    generated_state = st.session_state.generated_phrases.get(selected_intent)
    if generated_state:
        phrases_for_display = generated_state.get('phrases', [])
        st.caption(f"Last generated: {generated_state.get('generated_at', 'unknown time')}")
        st.dataframe(pd.DataFrame({selected_intent: phrases_for_display}), width='stretch')

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

    # ---- Export ----
    st.header("Export Data & Next Steps")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate Complete Excel Workbook", type="primary", key="excel_tab1"):
            with st.spinner("Creating Excel workbook..."):
                try:
                    output = generate_excel_workbook(
                        intent_names, intents, confusing_pairs, priority_actions,
                        phrase_confusion, intent_sim_matrix, all_phrases,
                        similarity_threshold, phrase_similarity_threshold,
                    )
                    st.download_button(
                        "Download Complete Excel Workbook", output,
                        "intent_analysis_complete.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='download-excel-tab1',
                    )
                    st.success("Excel workbook generated!")
                except Exception as e:
                    st.error(f"Error creating Excel file: {str(e)}")

    with col2:
        csv = st.session_state.data.to_csv(index=False)
        st.download_button("Download Modified Dataset", csv, "modified_intents.csv", "text/csv", key='download-csv-tab1')

    with col3:
        if st.button("Generate Full Report", key="report_tab1"):
            report = generate_text_report(intent_names, intents, confusing_pairs, phrase_confusion)
            st.download_button("Download Full Report", report, "intent_analysis_report.txt", "text/plain", key='download-report-tab1')

    # Checklist
    st.markdown("---")
    st.subheader("Next Steps Checklist")
    checklist = []
    if critical_pairs > 0:
        checklist.append("Address CRITICAL priority actions")
    if high_sim_pairs > 0:
        checklist.append("Review HIGH priority phrase-level conflicts")
    if phrase_confusion:
        checklist.append("Implement phrase-level actions")
    if confusing_pairs:
        checklist.append("Emphasize unique keywords")
    checklist += ["Generate new diverse phrases", "Export modified dataset", "Re-run analysis to verify improvements"]
    for item in checklist:
        st.markdown(f"- [ ] {item}")
