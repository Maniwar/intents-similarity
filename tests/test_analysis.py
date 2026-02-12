import pytest
import numpy as np
from unittest.mock import patch
from core.analysis import (
    _find_phrase_conflicts_brute,
    compute_intent_health,
    compute_tsne,
    PHRASE_SIMILARITY_MIN,
)


def _make_normalised(n, d=8):
    rng = np.random.RandomState(42)
    emb = rng.randn(n, d).astype('float32')
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


class TestFindPhraseConflictsBrute:
    def test_detects_identical_phrases(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype='float32')
        phrases = ['hello', 'hello']
        intents = ['a', 'b']
        conflicts, limited = _find_phrase_conflicts_brute(emb, phrases, intents, 1000, 256)
        assert len(conflicts) == 1
        assert conflicts[0]['Note'] == '⚠️ EXACT DUPLICATE'

    def test_respects_limit(self):
        emb = _make_normalised(20, 4)
        phrases = [f"phrase_{i}" for i in range(20)]
        intents = ['a'] * 10 + ['b'] * 10
        conflicts, limited = _find_phrase_conflicts_brute(emb, phrases, intents, 2, 256)
        assert len(conflicts) <= 2

    def test_ignores_same_intent(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype='float32')
        phrases = ['same', 'same']
        intents = ['a', 'a']  # same intent
        conflicts, _ = _find_phrase_conflicts_brute(emb, phrases, intents, 1000, 256)
        assert len(conflicts) == 0


class TestComputeIntentHealth:
    def test_basic_health_scores(self):
        intents = {
            'greeting': ['hello', 'hi', 'hey', 'good morning'],
            'farewell': ['bye', 'goodbye', 'see you', 'later'],
        }
        # Simulate embeddings: greeting cluster near [1,0], farewell near [0,1]
        emb = np.array([
            [0.95, 0.05], [0.90, 0.10], [0.85, 0.15], [0.92, 0.08],
            [0.05, 0.95], [0.10, 0.90], [0.15, 0.85], [0.08, 0.92],
        ], dtype='float32')
        phrase_to_intent = ['greeting'] * 4 + ['farewell'] * 4
        intent_names = ['greeting', 'farewell']

        records = compute_intent_health(intents, emb, phrase_to_intent, intent_names)
        assert len(records) == 2
        for r in records:
            assert 'Health Score' in r
            assert 0 <= r['Health Score'] <= 100
            assert r['Grade'] in ['🟢 Good', '🟡 Fair', '🔴 Needs Work']


class TestComputeTsne:
    def test_returns_2d(self):
        emb = _make_normalised(30, 8)
        coords = compute_tsne(emb, perplexity=5)
        assert coords.shape == (30, 2)

    def test_handles_small_n(self):
        emb = _make_normalised(5, 4)
        coords = compute_tsne(emb, perplexity=30)  # perplexity > n should be clamped
        assert coords.shape == (5, 2)
