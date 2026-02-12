from core.keywords import extract_keywords, analyze_keyword_overlap
import logging

logger = logging.getLogger(__name__)


def build_priority_actions(confusing_pairs, intents):
    actions = []
    for pair in confusing_pairs:
        sim_score = float(pair['Similarity'])
        if sim_score <= 0.9:
            continue
        intent_a = pair['Intent A']
        intent_b = pair['Intent B']
        kw = analyze_keyword_overlap(intents[intent_a], intents[intent_b])
        actions.append({
            'Priority': '🔴 CRITICAL',
            'Action': f"Merge or significantly differentiate '{intent_a}' and '{intent_b}'",
            'Reason': f"Extremely high similarity ({sim_score:.3f})",
            'Impact': 'High',
            'Common Keywords': ', '.join(kw['common_keywords'][:5]) if kw['common_keywords'] else 'None',
            'Suggested Fix': (
                f"Add unique keywords to each. For '{intent_a}': "
                f"{', '.join(kw['intent1_unique'][:3])}. For '{intent_b}': "
                f"{', '.join(kw['intent2_unique'][:3])}"
            ),
        })

    return actions


def build_phrase_issue_actions(confusing_pairs, phrase_confusion, intents):
    """Medium-priority actions based on phrase-level conflicts."""
    actions = []
    if not phrase_confusion:
        return actions

    phrase_issues = {}
    sample = phrase_confusion[:1000]
    for c in sample:
        key = (c['Intent'], c['Other Intent'])
        phrase_issues.setdefault(key, []).append(c)

    for (ia, ib), confs in sorted(phrase_issues.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        actions.append({
            'Priority': '🟡 HIGH',
            'Action': f"Review {len(confs)} phrases in '{ia}' similar to '{ib}'",
            'Reason': f"{len(confs)} phrases causing confusion",
            'Impact': 'Medium',
            'Common Keywords': 'See phrase-level details',
            'Suggested Fix': "Rephrase or move the most similar phrases",
        })
    return actions


def build_phrase_recommendations(phrase_confusion, sample_size=500):
    recs = []
    for confusion in phrase_confusion[:sample_size]:
        phrase = confusion['Phrase']
        current_intent = confusion['Intent']
        similar_phrase = confusion['Similar To']
        other_intent = confusion['Other Intent']
        similarity = float(confusion['Similarity'])

        if similarity > 0.95:
            action = "🔴 MOVE or DELETE"
            reason = "Almost identical to other intent"
        elif similarity > 0.9:
            action = "🟡 REPHRASE"
            reason = "Very similar - needs differentiation"
        else:
            action = "🟢 REVIEW"
            reason = "Moderately similar - consider rewording"

        kw_phrase = extract_keywords(phrase)
        kw_similar = extract_keywords(similar_phrase)
        common = set(kw_phrase) & set(kw_similar)
        unique_to_other = set(kw_similar) - common

        recs.append({
            'Action': action,
            'Your Phrase': phrase[:80] + '...' if len(phrase) > 80 else phrase,
            'Current Intent': current_intent,
            'Conflicts With Phrase': similar_phrase[:80] + '...' if len(similar_phrase) > 80 else similar_phrase,
            'Other Intent': other_intent,
            'Similarity': f"{similarity:.3f}",
            'Reason': reason,
            'Common Words': ', '.join(list(common)[:5]) if common else 'None',
            'Add to Current': ', '.join(list(unique_to_other)[:3]) if unique_to_other else 'N/A',
            'Suggestion': (
                f"Either move to '{other_intent}', delete, or add unique words like: "
                f"{', '.join(list(unique_to_other)[:3])}"
            ) if unique_to_other else "Consider removing one of these phrases",
        })
    return recs
