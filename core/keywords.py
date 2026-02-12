import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Multilingual stop words covering the top languages supported by the embedding models.
# English forms the base; common function words from Spanish, French, German, Portuguese,
# Italian, Dutch, and a handful of other major languages are included so keyword extraction
# degrades gracefully for non-English datasets.  Everything runs locally -- no NLTK download
# or network call required.
STOP_WORDS = frozenset({
    # English
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'can', 'want', 'need', 'please', 'would',
    'could', 'should', 'shall', 'will', 'may', 'might', 'must', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'don', 'now', 'also', 'well', 'like', 'get', 'got', 'go', 'going', 'went',
    'come', 'came', 'make', 'made', 'take', 'took', 'give', 'gave', 'say', 'said', 'tell',
    'told', 'know', 'knew', 'think', 'thought', 'see', 'saw', 'look', 'looked',
    # Spanish
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'en', 'y', 'o',
    'que', 'es', 'no', 'se', 'lo', 'con', 'por', 'para', 'su', 'al', 'como', 'pero', 'si',
    'me', 'te', 'le', 'nos', 'yo', 'tu', 'mi', 'este', 'esta', 'esto', 'ese', 'esa',
    'ser', 'estar', 'haber', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver', 'dar',
    'mas', 'ya', 'muy', 'sin', 'sobre', 'entre', 'cuando', 'todo', 'esta', 'son', 'tiene',
    # French
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'en', 'est', 'que', 'qui',
    'ne', 'pas', 'ce', 'se', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'je', 'tu', 'son',
    'sa', 'ses', 'au', 'aux', 'avec', 'pour', 'sur', 'dans', 'par', 'plus', 'ou', 'mais',
    'si', 'tout', 'bien', 'aussi', 'comme', 'peut', 'fait', 'etre', 'avoir', 'faire',
    # German
    'der', 'die', 'das', 'ein', 'eine', 'und', 'ist', 'ich', 'du', 'er', 'sie', 'es',
    'wir', 'ihr', 'nicht', 'den', 'dem', 'dem', 'von', 'zu', 'mit', 'auf', 'fur', 'an',
    'bei', 'nach', 'uber', 'auch', 'aber', 'oder', 'wenn', 'was', 'wie', 'hat', 'bin',
    'sind', 'war', 'wird', 'kann', 'muss', 'soll', 'noch', 'schon', 'nur', 'sehr',
    # Portuguese
    'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da', 'em', 'no', 'na', 'e', 'que',
    'para', 'com', 'por', 'se', 'mais', 'mas', 'como', 'ou', 'tem', 'ser', 'ter', 'foi',
    # Italian
    'il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'del', 'della', 'in', 'e', 'che', 'non',
    'un', 'una', 'per', 'con', 'su', 'sono', 'come', 'ma', 'anche', 'ha', 'ho', 'questo',
    # Dutch
    'de', 'het', 'een', 'van', 'en', 'in', 'is', 'dat', 'op', 'te', 'met', 'voor', 'niet',
    'zijn', 'er', 'maar', 'ook', 'aan', 'om', 'dit', 'die', 'dan', 'wat', 'nog', 'wel',
})


def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z\u00C0-\u024F]+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def extract_keywords_tfidf(intents, max_features=50, ngram_range=(1, 2)):
    """Use TF-IDF to find the most discriminative terms per intent.

    Returns a dict mapping intent name -> list of (term, score) tuples sorted
    descending by TF-IDF score.
    """
    intent_names = list(intents.keys())
    corpus = [' '.join(phrases) for phrases in intents.values()]

    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(STOP_WORDS),
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        result = {}
        for idx, intent in enumerate(intent_names):
            row = tfidf_matrix[idx].toarray().flatten()
            top_indices = row.argsort()[::-1]
            terms = [(feature_names[i], float(row[i])) for i in top_indices if row[i] > 0]
            result[intent] = terms[:max_features]
        return result
    except Exception:
        logger.exception("TF-IDF extraction failed, falling back to simple extraction")
        result = {}
        for intent, phrases in intents.items():
            all_kw = []
            for p in phrases:
                all_kw.extend(extract_keywords(p))
            counts = Counter(all_kw)
            result[intent] = [(w, c) for w, c in counts.most_common(max_features)]
        return result


def analyze_keyword_overlap(intent1_phrases, intent2_phrases):
    keywords1 = []
    for phrase in intent1_phrases:
        keywords1.extend(extract_keywords(phrase))
    keywords2 = []
    for phrase in intent2_phrases:
        keywords2.extend(extract_keywords(phrase))

    counter1 = Counter(keywords1)
    counter2 = Counter(keywords2)

    common_keywords = set(counter1.keys()) & set(counter2.keys())
    overlap_score = len(common_keywords) / max(len(set(keywords1)), len(set(keywords2)), 1)

    return {
        'common_keywords': sorted(common_keywords),
        'overlap_score': overlap_score,
        'intent1_unique': sorted(set(counter1.keys()) - common_keywords)[:10],
        'intent2_unique': sorted(set(counter2.keys()) - common_keywords)[:10],
    }
