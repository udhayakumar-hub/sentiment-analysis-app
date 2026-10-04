import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from src.preprocessor import TextPreprocessor

# Ensure VADER lexicon is downloaded
try:
    nltk.download("vader_lexicon", quiet=True)
except Exception:
    pass


class SentimentAnalyzer:
    """
    NLP Sentiment Analysis engine supporting:
    1. NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner)
    2. TextBlob (Pattern-based sentiment analyzer)
    """

    def __init__(self):
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception:
            nltk.download("vader_lexicon", quiet=True)
            self.vader = SentimentIntensityAnalyzer()
        self.preprocessor = TextPreprocessor()

    def analyze_vader(self, text: str) -> dict:
        """
        Analyzes sentiment using NLTK VADER.
        VADER works best on raw text (it captures punctuation, casing, emojis).
        """
        if not text or not text.strip():
            return self._empty_result("VADER")

        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]

        # Classification rule standard for VADER
        if compound >= 0.05:
            sentiment = "Positive"
            badge_color = "#10B981"  # Emerald Green
            emoji = "😊"
            confidence = round((compound) * 100, 1) if compound > 0 else 50.0
        elif compound <= -0.05:
            sentiment = "Negative"
            badge_color = "#EF4444"  # Red
            emoji = "😞"
            confidence = round(abs(compound) * 100, 1)
        else:
            sentiment = "Neutral"
            badge_color = "#F59E0B"  # Amber
            emoji = "😐"
            confidence = round((1.0 - abs(compound)) * 100, 1)

        return {
            "model": "NLTK VADER",
            "sentiment": sentiment,
            "emoji": emoji,
            "badge_color": badge_color,
            "compound_score": round(compound, 4),
            "polarity_score": round(compound, 4),
            "positive_score": round(scores["pos"], 4),
            "neutral_score": round(scores["neu"], 4),
            "negative_score": round(scores["neg"], 4),
            "confidence": min(100.0, max(50.0, confidence)),
            "details": {
                "Compound (Overall)": round(compound, 4),
                "Positive Ratio": f"{scores['pos'] * 100:.1f}%",
                "Neutral Ratio": f"{scores['neu'] * 100:.1f}%",
                "Negative Ratio": f"{scores['neg'] * 100:.1f}%",
            }
        }

    def analyze_textblob(self, text: str) -> dict:
        """
        Analyzes sentiment using TextBlob.
        Computes polarity (-1 to 1) and subjectivity (0 to 1).
        """
        if not text or not text.strip():
            return self._empty_result("TextBlob")

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.05:
            sentiment = "Positive"
            badge_color = "#10B981"
            emoji = "😊"
            confidence = round(polarity * 100, 1)
        elif polarity < -0.05:
            sentiment = "Negative"
            badge_color = "#EF4444"
            emoji = "😞"
            confidence = round(abs(polarity) * 100, 1)
        else:
            sentiment = "Neutral"
            badge_color = "#F59E0B"
            emoji = "😐"
            confidence = round((1.0 - abs(polarity)) * 100, 1)

        return {
            "model": "TextBlob",
            "sentiment": sentiment,
            "emoji": emoji,
            "badge_color": badge_color,
            "polarity_score": round(polarity, 4),
            "compound_score": round(polarity, 4),
            "subjectivity_score": round(subjectivity, 4),
            "positive_score": round(max(0.0, polarity), 4),
            "neutral_score": round(1.0 - abs(polarity), 4),
            "negative_score": round(abs(min(0.0, polarity)), 4),
            "confidence": min(100.0, max(50.0, confidence)),
            "details": {
                "Polarity (-1 to +1)": round(polarity, 4),
                "Subjectivity (0 to 1)": round(subjectivity, 4),
                "Subjectivity Type": "Opinion / Subjective" if subjectivity > 0.5 else "Factual / Objective"
            }
        }

    def extract_sentiment_words(self, text: str) -> dict:
        """
        Extracts individual positive and negative words from the text
        using VADER lexicon for educational breakdown.
        """
        pipeline = self.preprocessor.process_pipeline(text)
        tokens = pipeline["tokens"]
        pos_words = []
        neg_words = []

        for token in tokens:
            t_lower = token.lower()
            if t_lower in self.vader.lexicon:
                score = self.vader.lexicon[t_lower]
                if score > 0.5:
                    pos_words.append((token, score))
                elif score < -0.5:
                    neg_words.append((token, score))

        return {
            "positive_words": pos_words,
            "negative_words": neg_words,
            "pipeline": pipeline
        }

    def analyze(self, text: str, model_type: str = "NLTK VADER") -> dict:
        """
        Unified analysis entry point.
        """
        if model_type == "TextBlob":
            res = self.analyze_textblob(text)
        else:
            res = self.analyze_vader(text)

        # Include token breakdown & highlighted words
        breakdown = self.extract_sentiment_words(text)
        res["word_breakdown"] = breakdown
        res["pipeline"] = breakdown["pipeline"]
        return res

    def _empty_result(self, model_name: str) -> dict:
        return {
            "model": model_name,
            "sentiment": "Neutral",
            "emoji": "😐",
            "badge_color": "#F59E0B",
            "compound_score": 0.0,
            "polarity_score": 0.0,
            "positive_score": 0.0,
            "neutral_score": 1.0,
            "negative_score": 0.0,
            "confidence": 50.0,
            "details": {},
            "word_breakdown": {"positive_words": [], "negative_words": []},
            "pipeline": self.preprocessor.process_pipeline("")
        }
