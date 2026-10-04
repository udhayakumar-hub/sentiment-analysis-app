# 🧠 Sentiment Analysis App (NLP Web Application)

A modern, interactive Natural Language Processing (NLP) web application built with **Python**, **Streamlit**, and **NLTK / TextBlob**. The application analyzes and classifies user-provided text into **Positive**, **Negative**, or **Neutral** sentiment with real-time linguistic metrics, step-by-step token preprocessing pipeline visualization, session analytics, and export capabilities.

Designed specifically for college projects, NLP demonstrations, and student portfolios.

---

## 🌟 Key Features

1. **Rule-Based & Lexicon NLP Engines**:
   - **NLTK VADER** (*Valence Aware Dictionary and sEntiment Reasoner*): Optimized for social text, product reviews, capitalization heuristics, punctuation intensity (`!`), and negation flipping (`not good`).
   - **TextBlob Sentiment**: Pattern-based NLP computing polarity `[-1.0, +1.0]` and subjectivity `[0.0, 1.0]`.

2. **Interactive UI & Real-Time Classification**:
   - Large custom text area for entering reviews, sentences, feedback, or tweets.
   - Clean color-coded prediction badges (Emerald Green for Positive, Red for Negative, Amber for Neutral).
   - Real-time confidence score and interactive **Polarity Gauge Meter**.
   - Sentiment-bearing lexicon keywords detection (highlighting emotional words with scores).

3. **Curated Sample Texts**:
   - One-click testing with pre-loaded Positive, Negative, and Neutral reviews/statements across customer feedback, product ratings, movie reviews, and factual news.

4. **Educational NLP Preprocessing Pipeline**:
   - Step-by-step visual breakdown:
     1. Raw Text Input
     2. Cleaned & Normalized Text
     3. Lowercase Conversion
     4. Tokenization (word segmentation)
     5. Stopword Filtering (preserving negations)
     6. Lemmatization (WordNet base lemmas)

5. **Session Statistics & Analytics**:
   - Summary metric cards (Total Analyzed, Positive %, Neutral %, Negative %, Average Score).
   - Interactive Plotly charts: Sentiment Distribution Donut Chart and Sentiment Count Bar Chart.
   - Polarity progression timeline across user inputs.

6. **History & CSV Export**:
   - Maintains all analyzed texts with timestamps, predicted labels, polarity scores, and models used.
   - Search & filter functionality by sentiment class.
   - One-click CSV download of analysis history.

7. **Zero-API Dependency & Local Execution**:
   - Runs 100% locally on your machine without requiring paid APIs or internet connection for inference.

---

## 🛠️ Technology Stack

- **Frontend / Framework**: [Streamlit](https://streamlit.io/)
- **NLP Libraries**: [NLTK](https://www.nltk.org/) (VADER, Tokenize, Corpus, Lemmatizer) & [TextBlob](https://textblob.readthedocs.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
- **Interactive Visualizations**: [Plotly](https://plotly.com/python/)
- **Testing**: [Pytest](https://docs.pytest.org/)
- **Language**: Python 3.9+

---

## 📂 Project Structure

```text
sentiment-analysis-app/
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & presentation guide
├── .gitignore                 # Git ignore configuration
├── src/                       # Core NLP package
│   ├── __init__.py            # Package initialization
│   ├── preprocessor.py        # Text cleaning, tokenization, stopwords, lemmatization
│   ├── analyzer.py            # VADER and TextBlob sentiment classification logic
│   └── utils.py               # Pre-loaded sample texts, session history & analytics
└── tests/                     # Unit test suite
    ├── test_preprocessor.py   # Tests for NLP preprocessing pipeline
    └── test_analyzer.py       # Tests for sentiment classification models
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher installed on your system.
- Git (optional, for version control).

### Step 1: Clone or Open the Repository
```bash
git clone <your-repo-url>
cd "Sentiment Analysis App"
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```bash
# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Required NLTK Corpora (Automated or Manual)
The application automatically downloads required NLTK resources on first launch. If you wish to pre-download them manually:
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 💻 Running the Application

Launch the Streamlit web application by executing:
```bash
streamlit run app.py
```

The app will start and automatically open in your default browser at:
```
http://localhost:8501
```

---

## 🧪 Running Unit Tests

To run the automated test suite using `pytest`:
```bash
pytest -v
```

All unit tests for text cleaning, tokenization, negation handling, lemmatization, and sentiment classification should pass cleanly.

---

## 📖 How NLP Sentiment Analysis Works (College Presentation Guide)

### 1. What is Sentiment Analysis?
Sentiment Analysis (also known as *Opinion Mining*) is an NLP task that uses computational linguistics to determine the emotional polarity and subjective opinion expressed in human language.

### 2. The NLP Classification Workflow:
```
Raw Text 
  │
  ▼
1. Preprocessing (Cleaning, Lowercasing, Tokenization, Stopword Removal, Lemmatization)
  │
  ▼
2. Lexicon & Feature Extraction (Matching tokens against validated sentiment dictionaries)
  │
  ▼
3. Scoring & Heuristics (Computing valence, handling intensifiers, punctuation, and negations)
  │
  ▼
4. Polarity Output:
   - Compound Score >= +0.05  👉 Positive 😊
   - Compound Score <= -0.05  👉 Negative 😞
   - Score between -0.05 & +0.05 👉 Neutral 😐
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
