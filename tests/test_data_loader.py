import pytest
import pandas as pd
import io
from utils.data_loader import (
    load_intent_dataframe,
    build_intents_dict,
    validate_intents,
    find_cross_intent_duplicates,
    remove_cross_intent_duplicates,
)


def _make_csv(text, encoding='utf-8'):
    buf = io.BytesIO(text.encode(encoding))
    buf.name = "test.csv"
    buf.size = len(text)
    return buf


class TestLoadIntentDataframe:
    def test_auto_detect_utf8(self):
        csv = _make_csv("greeting,farewell\nhello,goodbye\nhi,bye\n")
        df, enc = load_intent_dataframe(csv, "Auto-detect")
        assert enc == 'utf-8'
        assert list(df.columns) == ['greeting', 'farewell']
        assert len(df) == 2

    def test_explicit_encoding(self):
        csv = _make_csv("a,b\n1,2\n", encoding='utf-8')
        df, enc = load_intent_dataframe(csv, "UTF-8")
        assert enc == 'utf-8'

    def test_none_file(self):
        df, enc = load_intent_dataframe(None, "UTF-8")
        assert df is None
        assert enc is None


class TestBuildIntentsDict:
    def test_basic(self):
        df = pd.DataFrame({'greeting': ['hello', 'hi', None], 'farewell': ['bye', None, None]})
        intents = build_intents_dict(df)
        assert 'greeting' in intents
        assert len(intents['greeting']) == 2
        assert len(intents['farewell']) == 1

    def test_skips_empty_columns(self):
        df = pd.DataFrame({'greeting': ['hello'], 'empty': [None]})
        intents = build_intents_dict(df)
        assert 'empty' not in intents


class TestValidateIntents:
    def test_warns_on_few_phrases(self):
        intents = {'a': ['only one']}
        warnings = validate_intents(intents)
        assert any('too few' in w['message'] for w in warnings)

    def test_warns_on_long_phrases(self):
        intents = {'a': ['x' * 600, 'normal', 'also normal']}
        warnings = validate_intents(intents)
        assert any('longer than 500' in w['message'] for w in warnings)

    def test_warns_on_duplicate_intent_names(self):
        intents = {'Greeting': ['hi'], 'greeting': ['hello']}
        warnings = validate_intents(intents)
        assert any('Duplicate intent names' in w['message'] for w in warnings)

    def test_warns_on_cross_duplicates(self):
        intents = {'a': ['hello', 'hi'], 'b': ['hello', 'bye']}
        warnings = validate_intents(intents)
        assert any('duplicate phrase' in w['message'] for w in warnings)


class TestFindCrossIntentDuplicates:
    def test_finds_duplicates(self):
        intents = {'a': ['hello', 'hi'], 'b': ['hello', 'bye']}
        dupes = find_cross_intent_duplicates(intents)
        assert len(dupes) == 1
        assert dupes[0]['Count'] == 2

    def test_no_duplicates(self):
        intents = {'a': ['hello'], 'b': ['goodbye']}
        dupes = find_cross_intent_duplicates(intents)
        assert len(dupes) == 0


class TestRemoveCrossIntentDuplicates:
    def test_removes_second_occurrence(self):
        df = pd.DataFrame({'a': ['hello', 'hi'], 'b': ['hello', 'bye']})
        intents = build_intents_dict(df)
        cleaned = remove_cross_intent_duplicates(df, intents)
        # 'hello' should remain in column 'a' but be removed from 'b'
        a_vals = cleaned['a'].dropna().tolist()
        b_vals = cleaned['b'].dropna().tolist()
        assert 'hello' in a_vals
        assert 'hello' not in b_vals
        assert 'bye' in b_vals
