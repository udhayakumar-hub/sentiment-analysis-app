import pandas as pd
import datetime

SAMPLE_TEXTS = {
    "🌟 Positive Reviews & Feedback": [
        "This product exceeded all my expectations! High quality, fast delivery, and truly exceptional customer service.",
        "An absolute masterpiece of modern cinema! The cinematography, acting, and background score are simply breathtaking.",
        "I love this new laptop! The display is gorgeous, performance is lightning fast, and battery life easily lasts all day.",
        "Outstanding learning experience! The professor explained complex NLP algorithms in a very intuitive and fun manner."
    ],
    "⚠️ Negative Reviews & Complaints": [
        "Worst customer service experience ever. The package arrived completely damaged and nobody replied to my emails.",
        "The application constantly crashes on startup. It is extremely slow, buggy, and completely unusable.",
        "Very disappointed with the food quality. It was cold, overpriced, and the staff was very rude.",
        "I regret buying this gadget. It stopped working after two days and the company refused to give a refund."
    ],
    "⚖️ Neutral & Factual Statements": [
        "The international conference on artificial intelligence is scheduled for October 15th at 9:00 AM.",
        "The package was shipped on Tuesday and contains three standard electrical replacement cables.",
        "The library has updated its operational hours for the upcoming semester examination period.",
        "Water boils at 100 degrees Celsius under standard atmospheric pressure conditions."
    ]
}


def add_to_history(session_state, text: str, sentiment: str, score: float, model: str):
    """Adds a new prediction entry to the session history."""
    if "history" not in session_state:
        session_state.history = []

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {
        "Time": timestamp,
        "Text": text,
        "Sentiment": sentiment,
        "Polarity Score": score,
        "Model": model
    }
    # Prepend newest first
    session_state.history.insert(0, entry)


def get_history_df(session_state) -> pd.DataFrame:
    """Returns the history list as a clean pandas DataFrame."""
    if "history" not in session_state or not session_state.history:
        return pd.DataFrame(columns=["Time", "Text", "Sentiment", "Polarity Score", "Model"])
    return pd.DataFrame(session_state.history)


def calculate_statistics(history_df: pd.DataFrame) -> dict:
    """Calculates statistics from the current session history."""
    if history_df.empty:
        return {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "pos_pct": 0.0,
            "neg_pct": 0.0,
            "neu_pct": 0.0,
            "avg_score": 0.0
        }

    total = len(history_df)
    counts = history_df["Sentiment"].value_counts().to_dict()
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    neu = counts.get("Neutral", 0)
    avg_score = round(history_df["Polarity Score"].mean(), 3)

    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "pos_pct": round((pos / total) * 100, 1),
        "neg_pct": round((neg / total) * 100, 1),
        "neu_pct": round((neu / total) * 100, 1),
        "avg_score": avg_score
    }
