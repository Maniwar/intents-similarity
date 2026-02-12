import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

ENCODING_MAP = {
    "Auto-detect": None,
    "UTF-8": "utf-8",
    "Windows-1252 (Excel)": "cp1252",
    "Latin-1": "latin-1",
    "UTF-16": "utf-16",
}

AUTO_DETECT_ENCODINGS = ['utf-8', 'cp1252', 'latin-1', 'utf-16']


def load_intent_dataframe(uploaded_file, encoding_option):
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


def build_intents_dict(df):
    intents = {}
    for col in df.columns:
        phrases = df[col].dropna().tolist()
        phrases = [str(p).strip() for p in phrases if str(p).strip() and str(p).strip().lower() != 'nan']
        if phrases:
            intents[col] = phrases
    return intents


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_intents(intents):
    """Run quality checks on the parsed intent data.

    Returns a list of warning dicts: {level: 'error'|'warning'|'info', message: str}
    """
    warnings = []

    # Check intents with very few phrases
    for name, phrases in intents.items():
        if len(phrases) < 3:
            warnings.append({
                'level': 'warning',
                'message': f"Intent '{name}' has only {len(phrases)} phrase(s) -- too few for meaningful analysis.",
            })

    # Check for very long phrases (likely data errors)
    for name, phrases in intents.items():
        long = [p for p in phrases if len(p) > 500]
        if long:
            warnings.append({
                'level': 'warning',
                'message': f"Intent '{name}' has {len(long)} phrase(s) longer than 500 characters -- possible data error.",
            })

    # Check for non-text content (URLs, HTML, numbers-only)
    url_pattern = re.compile(r'^https?://')
    html_pattern = re.compile(r'<[^>]+>')
    for name, phrases in intents.items():
        urls = sum(1 for p in phrases if url_pattern.match(p))
        html = sum(1 for p in phrases if html_pattern.search(p))
        numeric = sum(1 for p in phrases if p.replace('.', '').replace(',', '').isdigit())
        issues = urls + html + numeric
        if issues:
            warnings.append({
                'level': 'warning',
                'message': f"Intent '{name}' has {issues} non-text phrase(s) (URLs/HTML/numbers).",
            })

    # Check for case-insensitive duplicate intent names
    lower_names = {}
    for name in intents:
        key = name.strip().lower()
        lower_names.setdefault(key, []).append(name)
    for key, names in lower_names.items():
        if len(names) > 1:
            warnings.append({
                'level': 'error',
                'message': f"Duplicate intent names (case-insensitive): {', '.join(names)}",
            })

    # Cross-intent duplicate phrases
    from collections import Counter
    all_phrases = []
    for phrases in intents.values():
        all_phrases.extend(phrases)
    dup_count = len(all_phrases) - len(set(all_phrases))
    if dup_count > 0:
        warnings.append({
            'level': 'warning',
            'message': f"Found {dup_count} duplicate phrase(s) across intents. This may inflate similarity scores.",
        })

    return warnings


def find_cross_intent_duplicates(intents):
    """Return a list of {phrase, intents, count} for phrases that appear in multiple intents."""
    phrase_to_intents = {}
    for name, phrases in intents.items():
        for p in phrases:
            phrase_to_intents.setdefault(p, []).append(name)

    duplicates = []
    for phrase, intent_list in phrase_to_intents.items():
        if len(intent_list) > 1:
            duplicates.append({
                'Phrase': phrase[:80] + '...' if len(phrase) > 80 else phrase,
                'Count': len(intent_list),
                'Appears In': ', '.join(intent_list),
            })
    duplicates.sort(key=lambda x: x['Count'], reverse=True)
    return duplicates


def remove_cross_intent_duplicates(df, intents, keep='first'):
    """Remove phrases that appear in more than one intent column, keeping only the first occurrence.

    Returns a cleaned DataFrame.
    """
    seen = set()
    cleaned = df.copy()
    for col in cleaned.columns:
        vals = cleaned[col].tolist()
        new_vals = []
        for v in vals:
            if pd.isna(v):
                new_vals.append(v)
                continue
            s = str(v).strip()
            if s == '' or s.lower() == 'nan':
                new_vals.append(None)
                continue
            if s in seen:
                new_vals.append(None)
            else:
                seen.add(s)
                new_vals.append(v)
        cleaned[col] = new_vals

    # Compact: shift non-null values up in each column
    for col in cleaned.columns:
        non_null = cleaned[col].dropna().tolist()
        padded = non_null + [None] * (len(cleaned) - len(non_null))
        cleaned[col] = padded

    # Drop rows that are entirely null
    cleaned = cleaned.dropna(how='all').reset_index(drop=True)
    return cleaned
