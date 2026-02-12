import streamlit as st
import pandas as pd


def render_metric_row(metrics):
    """Render a row of st.metric widgets.

    *metrics* is a list of dicts with keys: label, value, and optional delta / delta_color.
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            kwargs = {'label': m['label'], 'value': m['value']}
            if 'delta' in m:
                kwargs['delta'] = m['delta']
            if 'delta_color' in m:
                kwargs['delta_color'] = m['delta_color']
            st.metric(**kwargs)


def render_data_quality_warnings(warnings):
    """Display data validation warnings using the appropriate Streamlit widget."""
    for w in warnings:
        if w['level'] == 'error':
            st.error(w['message'])
        elif w['level'] == 'warning':
            st.warning(w['message'])
        else:
            st.info(w['message'])


def filter_phrase_conflicts(phrase_conflicts_all, min_similarity, intent_filter="All Intents"):
    """Return a filtered list of phrase conflict dicts."""
    filtered = []
    for c in phrase_conflicts_all:
        if c['Similarity'] < min_similarity:
            continue
        if intent_filter != "All Intents":
            if c['Intent'] != intent_filter and c['Other Intent'] != intent_filter:
                continue
        filtered.append(c.copy())
    return filtered
