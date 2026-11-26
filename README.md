# Elevvo NLP Internship Portfolio 🚀

[![CI](https://github.com/Eslammohammedd/Elevvo-NLP-Internship/actions/workflows/ci.yml/badge.svg)](https://github.com/Eslammohammedd/Elevvo-NLP-Internship/actions/workflows/ci.yml)

Welcome to my **Natural Language Processing (NLP) Internship Portfolio**! This repository showcases the advanced NLP models and pipelines I developed during my internship at Elevvo. The objective of this training was to build end-to-end machine learning workflows for text analysis, ranging from classical ML techniques to modern Transformer-based architectures.

---

## 📂 Internship Projects

I successfully completed and standardized **8 NLP projects** (Tasks 1, 2, 3, 4, 5, 6, 7, and 8). Each project is modularized into a professional layout (`data/`, `notebooks/`, `src/`, `tests/`) with automated result generation to ensure high reproducibility and clean code standards.

---

### 1️⃣ [Sentiment Analysis on Product Reviews](./Task_1_Sentiment_Analysis)
- **Objective:** Classify IMDb movie reviews as positive or negative.
- **Dataset:** IMDb (25,000 train / 25,000 test samples).
- **Concepts:** TF-IDF, CountVectorizer, NLTK Preprocessing, Logistic Regression, Naive Bayes.
- **Best Result:** Logistic Regression — **85.3% Accuracy**, **85.1% F1-Score**.
- **Highlights:** Custom lemmatization pipeline; Word Cloud & model comparison visualizations auto-generated.

---

### 2️⃣ [News Category Classification](./Task_2_News_Classification)
- **Objective:** Categorize news articles into World, Sports, Business, or Sci/Tech.
- **Dataset:** AG News (120K train / 7.6K test samples).
- **Concepts:** Multiclass Classification, Linear SVM, Logistic Regression, Random Forest.
- **Best Result:** Linear SVM — **88.1% Accuracy**, **88.1% F1-Score**.
- **Highlights:** Competitive 4-class categorization; Category word clouds auto-generated.

---

### 3️⃣ [Fake News Detection](./Task_3_Fake_News)
- **Objective:** Detect whether a political statement is real or fake.
- **Dataset:** LIAR dataset (PolitiFact benchmark, binary-mapped).
- **Concepts:** Binary Classification, N-gram feature engineering (tri-grams).
- **Best Result:** Logistic Regression — **62.5% Accuracy**, **60.2% Macro F1-Score**.
- **Highlights:** Word Clouds per class, discriminating feature bar charts, and feature importance analysis.

---

### 4️⃣ [Named Entity Recognition from News](./Task_4_NER)
- **Objective:** Identify and classify named entities (PER, ORG, GPE, MONEY, etc.) in news text.
- **Models:** spaCy `en_core_web_sm` vs `en_core_web_md` and custom Rule-Based engine.
- **Concepts:** Rule-based NLP, Model-based NER, Information Extraction, spaCy pipelines.
- **Highlights:** Interactive **displaCy** entity visualization; Comparative analysis of model size vs. accuracy.

---

### 5️⃣ [Topic Modeling on News Articles](./Task_5_Topic_Modeling)
- **Objective:** Discover hidden themes in a news corpus using unsupervised NLP.
- **Dataset:** Custom 25-article news sample (5 topics).
- **Concepts:** LDA (Gensim), NMF (scikit-learn), Coherence Scoring, TF-IDF.
- **Highlights:** LDA & NMF model comparison; topic-word distribution charts; pyLDAvis-style exports.

---

### 6️⃣ [Question Answering with Transformers](./Task_6_Question_Answering)
- **Objective:** Build a system that answers questions based on a given context or passage.
- **Dataset:** SQuAD v1.1 benchmark (100 sample records evaluated).
- **Concepts:** Hugging Face Transformers, Span Extraction, Pre-trained Models.
- **Best Result:** DistilBERT (`distilled-squad`) — **57.0% Exact Match**, **74.2% F1-Score**.
- **Highlights:** Interactive Streamlit web UI; Automated evaluation of Exact Match and F1.

---

### 7️⃣ [Text Summarization Using Transformers](./Task_7_Text_Summarization)
- **Objective:** Automatically shrink long CNN/DailyMail news articles into concise summaries.
- **Dataset:** CNN/DailyMail (50 test articles evaluated).
- **Model:** Hugging Face `t5-small` (Abstractive) vs. TextRank (Extractive).
- **Best Result:** T5-Small — **ROUGE-1: 0.317 / ROUGE-2: 0.127 / ROUGE-L: 0.237**.
- **Highlights:** Side-by-side comparison pipeline; ROUGE scoring framework; length distribution analysis.

---

### 8️⃣ [Resume Screening Using NLP](./Task_8_Resume_Screening)
- **Objective:** Screen and rank candidate resumes against job descriptions.
- **Dataset:** Unstructured Resume texts and Job Descriptions.
- **Concepts:** SentenceTransformers, Cosine Similarity, Semantic Search, Named Entity Recognition (spaCy).
- **Best Result:** `all-mpnet-base-v2` — **~0.76 Avg Top-1 Matching Score**.
- **Highlights:** Interactive Streamlit matching application; Entity extraction of candidate skills.

---

## 🛠️ Technology Stack

| Category               | Libraries / Tools                                          |
|------------------------|------------------------------------------------------------|
| **Data & Manipulation** | Pandas, NumPy, Hugging Face Datasets                     |
| **Classical & Rule NLP**| NLTK, Scikit-Learn, Gensim, Regular Expressions (re)     |
| **Entity Extraction**   | **spaCy** (SM & MD models), displaCy                     |
| **Deep Learning / LLMs**| PyTorch, Transformers (Hugging Face), T5                 |
| **Visualization**       | Matplotlib, Seaborn, WordCloud                           |
| **Evaluation**          | ROUGE (rouge-score), Coherence (Gensim), F1/Accuracy    |
| **Persistence**         | Joblib (model serialization), JSON/CSV (results export)  |

---

## 🚀 How to Run

Each project is self-contained. To explore any project:

```bash
# 1. Navigate into the project folder
cd Task_1_Sentiment_Analysis   # (or Task_2, Task_3, Task_4, Task_5, Task_6, Task_7, Task_8)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete training & evaluation pipeline
python src/main.py

# 4. (Optional) Generate standardized results (CSV + JSON)
python src/generate_results.py

# 5. (Optional) Run unit tests
python -m pytest tests/ -v
```

Alternatively, explore the detailed Exploratory Data Analysis in the `notebooks/` folder.

---

## 📊 Results Summary

| Task | Best Model | Metric | Score |
|------|-----------|--------|-------|
| Task 1 — Sentiment Analysis | Logistic Regression (TF-IDF) | Accuracy | **85.3%** |
| Task 2 — News Classification | Linear SVM | Accuracy | **88.1%** |
| Task 3 — Fake News Detection | Logistic Regression | Accuracy | **62.5%** |
| Task 4 — NER (Bonus) | spaCy (MD model) | Accuracy | **~92%** |
| Task 5 — Topic Modeling | LDA + NMF | Qualitative | 5 topics |
| Task 6 — Question Answering | DistilBERT | F1-Score | **74.2%** |
| Task 7 — Text Summarization | T5-Small | ROUGE-L | **0.237** |
| Task 8 — Resume Screening | all-mpnet-base-v2 | Top-1 Score | **~0.76** |

---

## 🧪 Testing

Run each task's tests from its own folder so `src/` imports resolve correctly:

```bash
cd Task_1_Sentiment_Analysis
python -m pytest tests -q
```

Repeat for any task folder. CI runs the lightweight suites (Tasks 1–5 and 7) on every push and pull request. Tasks 6 and 8 load transformer models and are verified manually to keep CI fast.

## 📦 Data & Models

Datasets are generated/downloaded with `download_data.py`. Trained models are written to each task's `models/` folder by its `main.py`; regenerate them locally instead of relying on committed binaries. See [docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md) for exact commands and the list of ignored artifacts.

## 📄 License

MIT — see [LICENSE](LICENSE).

---

*Elevvo NLP Internship — Certificate of Completion — 2026*
