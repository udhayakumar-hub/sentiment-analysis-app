import pytest
from src.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_clean_text(preprocessor):
    raw = "<p>Check this link https://example.com #awesome @friend!</p>"
    cleaned = preprocessor.clean_text(raw)
    assert "<p>" not in cleaned
    assert "https://example.com" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "awesome" in cleaned


def test_to_lowercase(preprocessor):
    assert preprocessor.to_lowercase("NLP IS AMAZING") == "nlp is amazing"
    assert preprocessor.to_lowercase("") == ""


def test_tokenize(preprocessor):
    tokens = preprocessor.tokenize("Natural Language Processing is fun!")
    assert "Natural" in tokens or "natural" in [t.lower() for t in tokens]
    assert "Processing" in tokens or "processing" in [t.lower() for t in tokens]
    assert len(tokens) >= 5


def test_remove_stopwords(preprocessor):
    tokens = ["this", "is", "a", "great", "movie"]
    filtered = preprocessor.remove_stopwords(tokens, keep_negations=True)
    assert "this" not in filtered
    assert "is" not in filtered
    assert "great" in filtered
    assert "movie" in filtered


def test_remove_stopwords_preserves_negations(preprocessor):
    tokens = ["not", "good", "at", "all"]
    filtered = preprocessor.remove_stopwords(tokens, keep_negations=True)
    assert "not" in filtered
    assert "good" in filtered


def test_lemmatize(preprocessor):
    tokens = ["running", "cats", "better"]
    lemmas = preprocessor.lemmatize(tokens)
    assert "cat" in lemmas or "cats" in lemmas
    assert len(lemmas) == 3


def test_process_pipeline(preprocessor):
    res = preprocessor.process_pipeline("<p>The user did not like the experience!</p>")
    assert res["original_text"] == "<p>The user did not like the experience!</p>"
    assert res["token_count"] > 0
    assert "not" in res["filtered_tokens"]
