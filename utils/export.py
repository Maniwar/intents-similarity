import pandas as pd
import numpy as np
import json
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


def generate_excel_workbook(
    intent_names, intents, confusing_pairs, priority_actions,
    phrase_confusion, intent_sim_matrix, all_phrases,
    similarity_threshold, phrase_similarity_threshold,
):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_data = {
            'Metric': [
                'Total Intents', 'Total Phrases', 'Unique Phrases',
                'Confusing Intent Pairs', 'Total Phrase Conflicts',
                'Intent Threshold Used', 'Phrase Threshold Used',
            ],
            'Value': [
                len(intent_names),
                sum(len(p) for p in intents.values()),
                len(set(all_phrases)),
                len(confusing_pairs),
                len(phrase_confusion) if phrase_confusion else 0,
                similarity_threshold,
                phrase_similarity_threshold,
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        if confusing_pairs:
            pd.DataFrame(confusing_pairs).to_excel(writer, sheet_name='Confusing_Intent_Pairs', index=False)

        if priority_actions:
            pd.DataFrame(priority_actions).to_excel(writer, sheet_name='Priority_Actions', index=False)

        if phrase_confusion:
            pd.DataFrame(phrase_confusion).to_excel(writer, sheet_name='All_Phrase_Conflicts', index=False)

        intent_sim_df = pd.DataFrame(intent_sim_matrix, columns=intent_names, index=intent_names)
        intent_sim_df.to_excel(writer, sheet_name='Intent_Similarity_Matrix')

    output.seek(0)
    return output


def generate_search_excel(search_query, search_threshold, results_df, intent_dist_df, confidence, search_results):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_data = {
            'Metric': [
                'Search Query', 'Similarity Threshold', 'Total Results',
                'Unique Intents', 'Top Match Similarity', 'Top Match Intent',
                'Confidence Level',
            ],
            'Value': [
                search_query[:100],
                search_threshold,
                len(results_df),
                len(results_df['Intent'].unique()),
                f"{search_results[0]['similarity']:.3f}" if search_results else "N/A",
                search_results[0]['intent'] if search_results else "N/A",
                f"{confidence:.0%}" if confidence is not None else "N/A",
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        if intent_dist_df is not None and not intent_dist_df.empty:
            intent_dist_df.to_excel(writer, sheet_name='Intent_Distribution', index=False)

        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        results_df.head(10).to_excel(writer, sheet_name='Top_10_Matches', index=False)

    output.seek(0)
    return output


def generate_search_json(search_query, search_threshold, results_df, search_results, dominant_intent):
    return json.dumps({
        'query': search_query,
        'threshold': search_threshold,
        'timestamp': pd.Timestamp.now().isoformat(),
        'summary': {
            'total_results': len(results_df),
            'unique_intents': int(results_df['Intent'].nunique()),
            'top_similarity': float(search_results[0]['similarity']) if search_results else None,
            'recommended_intent': dominant_intent,
        },
        'results': results_df.to_dict(orient='records'),
    }, indent=2)


def generate_text_report(intent_names, intents, confusing_pairs, phrase_confusion):
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
        report += f"\n- {pair['Intent A']} <-> {pair['Intent B']}: {pair['Similarity']}\n"

    report += """
## Recommended Next Steps
1. Address CRITICAL priority items immediately
2. Review HIGH priority phrase-level conflicts
3. Add unique keywords from keyword analysis
4. Generate diverse phrases for low-diversity intents
5. Re-run analysis to validate improvements
"""
    return report
