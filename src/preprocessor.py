import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


class TextPreprocessor:
    """
    Educational NLP Text Preprocessor.
    Implements standard text cleaning, tokenization, stopword removal, and lemmatization.
    """

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """
        Cleans raw input text:
        - Removes URLs and HTML tags
        - Removes special characters / excessive punctuation
        - Normalizes whitespace
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions and hashtags symbols (preserving words)
        text = re.sub(r'[@#]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def to_lowercase(self, text: str) -> str:
        """Converts text to lowercase."""
        return text.lower() if text else ""

    def tokenize(self, text: str) -> list:
        """
        Tokenizes text into individual words/tokens.
        Uses NLTK word_tokenize with fallback to regex word splitting.
        """
        if not text:
            return []
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = re.findall(r'\b\w+\b', text)
        # Keep only alphanumeric tokens
        return [tok for tok in tokens if any(c.isalnum() for c in tok)]

    def remove_stopwords(self, tokens: list, keep_negations: bool = True) -> list:
        """
        Removes common english stopwords.
        Optionally preserves important negation words (not, no, never, etc.)
        which are crucial for sentiment analysis.
        """
        if not tokens:
            return []

        negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', "n't", 'hardly', 'barely', 'scarcely', 'cannot'}
        custom_stopwords = self.stop_words - negation_words if keep_negations else self.stop_words

        return [word for word in tokens if word.lower() not in custom_stopwords and word not in string.punctuation]

    def lemmatize(self, tokens: list) -> list:
        """
        Reduces tokens to their base dictionary lemma (e.g., 'running' -> 'running' / 'better' -> 'good' with pos tags or default noun).
        """
        if not tokens:
            return []
        return [self.lemmatizer.lemmatize(token.lower()) for token in tokens]

    def process_pipeline(self, raw_text: str) -> dict:
        """
        Executes the complete preprocessing pipeline step-by-step
        and returns intermediate representations for educational visualization.
        """
        cleaned = self.clean_text(raw_text)
        lowercased = self.to_lowercase(cleaned)
        tokens = self.tokenize(lowercased)
        filtered_tokens = self.remove_stopwords(tokens, keep_negations=True)
        lemmatized_tokens = self.lemmatize(filtered_tokens)
        processed_text = " ".join(lemmatized_tokens)

        return {
            "original_text": raw_text,
            "cleaned_text": cleaned,
            "lowercased_text": lowercased,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "processed_text": processed_text,
            "token_count": len(tokens),
            "filtered_count": len(filtered_tokens),
            "removed_stopwords_count": len(tokens) - len(filtered_tokens)
        }
