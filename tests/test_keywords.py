import pytest
from core.keywords import extract_keywords, analyze_keyword_overlap, extract_keywords_tfidf


class TestExtractKeywords:
    def test_removes_stop_words(self):
        kw = extract_keywords("I want to book a flight")
        assert 'want' not in kw
        assert 'book' in kw
        assert 'flight' in kw

    def test_removes_short_words(self):
        kw = extract_keywords("go to NY")
        assert 'go' not in kw  # stop word
        # "NY" is only 2 chars -- should be excluded
        assert all(len(w) > 2 for w in kw)

    def test_handles_empty_string(self):
        assert extract_keywords("") == []

    def test_handles_unicode(self):
        kw = extract_keywords("reservar un vuelo")
        # "un" is a stop word (Spanish), "reservar" and "vuelo" should survive
        assert 'reservar' in kw
        assert 'vuelo' in kw


class TestAnalyzeKeywordOverlap:
    def test_basic_overlap(self):
        phrases1 = ["book a flight", "reserve a flight"]
        phrases2 = ["cancel a flight", "delete a flight"]
        result = analyze_keyword_overlap(phrases1, phrases2)
        assert 'flight' in result['common_keywords']
        assert result['overlap_score'] > 0

    def test_no_overlap(self):
        phrases1 = ["hello there"]
        phrases2 = ["goodbye forever"]
        result = analyze_keyword_overlap(phrases1, phrases2)
        assert len(result['common_keywords']) == 0


class TestExtractKeywordsTfidf:
    def test_returns_terms_for_each_intent(self):
        intents = {
            "greeting": ["hello there", "good morning everyone"],
            "farewell": ["goodbye friend", "see you later"],
        }
        result = extract_keywords_tfidf(intents)
        assert "greeting" in result
        assert "farewell" in result
        assert len(result["greeting"]) > 0
        assert len(result["farewell"]) > 0

    def test_terms_are_tuples(self):
        intents = {"test": ["alpha beta gamma", "delta epsilon"]}
        result = extract_keywords_tfidf(intents)
        for term, score in result["test"]:
            assert isinstance(term, str)
            assert isinstance(score, float)
