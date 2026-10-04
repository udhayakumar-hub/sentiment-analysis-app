import pytest
from src.analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


def test_positive_sentiment_vader(analyzer):
    text = "This product is fantastic, outstanding, and absolutely wonderful!"
    res = analyzer.analyze(text, model_type="NLTK VADER")
    assert res["sentiment"] == "Positive"
    assert res["compound_score"] > 0.05
    assert res["confidence"] > 50.0


def test_negative_sentiment_vader(analyzer):
    text = "This service was awful, terrible, and completely unacceptable."
    res = analyzer.analyze(text, model_type="NLTK VADER")
    assert res["sentiment"] == "Negative"
    assert res["compound_score"] < -0.05
    assert res["confidence"] > 50.0


def test_neutral_sentiment_vader(analyzer):
    text = "The package contains three standard cables and an instruction manual."
    res = analyzer.analyze(text, model_type="NLTK VADER")
    assert res["sentiment"] == "Neutral"
    assert -0.1 <= res["compound_score"] <= 0.1


def test_textblob_analyzer(analyzer):
    text = "I love this wonderful application!"
    res = analyzer.analyze(text, model_type="TextBlob")
    assert res["sentiment"] == "Positive"
    assert res["polarity_score"] > 0.0
    assert "subjectivity_score" in res


def test_empty_string_handling(analyzer):
    res = analyzer.analyze("", model_type="NLTK VADER")
    assert res["sentiment"] == "Neutral"
    assert res["compound_score"] == 0.0


def test_word_breakdown_extraction(analyzer):
    text = "The food was great and delicious, but the service was terrible."
    breakdown = analyzer.extract_sentiment_words(text)
    pos_words = [w[0].lower() for w in breakdown["positive_words"]]
    neg_words = [w[0].lower() for w in breakdown["negative_words"]]
    assert "great" in pos_words or "delicious" in pos_words
    assert "terrible" in neg_words
