"""
generate_results.py — Task 5: Topic Modeling on News Articles
=============================================================
Loads the saved LDA & NMF models from models/ and generates a detailed,
structured results report including:
  - Topic-word distributions for both models
  - Per-document dominant-topic assignment   → data/results.csv
  - Coherence score evaluation
  - JSON summary                             → data/results_summary.json

Usage:
    python src/generate_results.py
"""

import os
import sys
import json
import pickle
import pandas as pd
from datetime import datetime

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── path setup ──────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from gensim import corpora, models as gensim_models
from preprocess import download_nltk_resources, preprocess_corpus
from evaluate  import compute_coherence, print_topics, print_coherence

# ── config ──────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_CSV  = os.path.join(DATA_DIR, "results.csv")
SUMMARY_JSON = os.path.join(DATA_DIR, "results_summary.json")
N_WORDS      = 10


# ── helpers ──────────────────────────────────────────────────────────────────
def _sep(char="=", n=60):
    print(char * n)


def _check_file(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            "  → Run `python src/main.py` first to train and save the models."
        )


def _load_lda_model():
    lda_path  = os.path.join(MODELS_DIR, "lda_model.gensim")
    dict_path = os.path.join(MODELS_DIR, "dictionary.gensim")
    _check_file(lda_path,  "LDA model")
    _check_file(dict_path, "Gensim dictionary")
    lda_model  = gensim_models.LdaModel.load(lda_path)
    dictionary = corpora.Dictionary.load(dict_path)
    return lda_model, dictionary


def _load_nmf_artifacts():
    nmf_path   = os.path.join(MODELS_DIR, "nmf_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    _check_file(nmf_path,   "NMF model")
    _check_file(tfidf_path, "TF-IDF vectorizer")
    with open(nmf_path, "rb") as f:
        nmf_model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    return nmf_model, tfidf_vectorizer


def _get_lda_topics(lda_model, n_words: int = 10) -> list:
    topics = []
    for idx, topic in lda_model.print_topics(num_words=n_words):
        words = [w.split('"')[1] for w in topic.split('+')]
        topics.append({"topic_id": idx, "words": [w.strip() for w in words]})
    return topics


def _get_nmf_topics(nmf_model, tfidf_vectorizer, n_words: int = 10) -> list:
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for idx, component in enumerate(nmf_model.components_):
        top_indices = component.argsort()[-n_words:][::-1]
        words = [feature_names[i] for i in top_indices]
        topics.append({"topic_id": idx, "words": words})
    return topics


def _get_dominant_topics_lda(lda_model, dictionary, tokenized_docs) -> list:
    corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    dominant = []
    for bow in corpus_bow:
        topic_dist = lda_model.get_document_topics(bow)
        best = max(topic_dist, key=lambda x: x[1]) if topic_dist else (0, 0.0)
        dominant.append({"topic_id": best[0], "probability": round(best[1], 4)})
    return dominant


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    _sep()
    print("  TASK 5 — TOPIC MODELING  |  Results Reporter")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep()

    # [1] NLTK resources
    print("\n[1] Downloading NLTK resources...")
    download_nltk_resources()

    # [2] Load dataset
    data_path = os.path.join(DATA_DIR, "sample_news.csv")
    _check_file(data_path, "Dataset (sample_news.csv)")
    print(f"[2] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"    Documents : {len(df)}")
    print(f"    Categories: {df['category'].unique().tolist()}")

    raw_texts = df["text"].tolist()

    # [3] Preprocess
    print("\n[3] Preprocessing text...")
    tokenized_docs = preprocess_corpus(raw_texts)
    avg_tokens = sum(len(d) for d in tokenized_docs) / len(tokenized_docs)
    print(f"    Avg tokens/doc (post-cleaning): {avg_tokens:.1f}")

    # [4] Load saved models
    print("\n[4] Loading saved models from models/...")
    lda_model, dictionary = _load_lda_model()
    nmf_model, tfidf_vec  = _load_nmf_artifacts()
    print("    [OK] LDA model loaded")
    print("    [OK] NMF model loaded")

    # [5] Topic-word distributions
    _sep()
    print("  LDA — TOP WORDS PER TOPIC")
    _sep()
    lda_topics = _get_lda_topics(lda_model, N_WORDS)
    print_topics(lda_topics, "LDA (Gensim)")

    _sep()
    print("  NMF — TOP WORDS PER TOPIC")
    _sep()
    nmf_topics = _get_nmf_topics(nmf_model, tfidf_vec, N_WORDS)
    print_topics(nmf_topics, "NMF (scikit-learn)")

    # [6] Coherence score
    print("[6] Computing LDA coherence score (U_Mass)...")
    lda_coherence = compute_coherence(lda_model, tokenized_docs, dictionary)
    print_coherence(lda_coherence, "LDA")

    # [7] Document-topic assignment
    print("[7] Building per-document topic assignment...")
    dominant_topics = _get_dominant_topics_lda(lda_model, dictionary, tokenized_docs)
    df["dominant_topic"]       = [d["topic_id"]    for d in dominant_topics]
    df["topic_probability"]    = [d["probability"] for d in dominant_topics]
    df["dominant_topic_words"] = [
        " | ".join(lda_topics[d["topic_id"]]["words"][:5])
        for d in dominant_topics
    ]

    # NMF doc-topic scores
    tfidf_matrix = tfidf_vec.transform(raw_texts)
    nmf_scores   = nmf_model.transform(tfidf_matrix)
    df["nmf_dominant_topic"]  = nmf_scores.argmax(axis=1)
    df["nmf_topic_words"]     = [
        " | ".join(nmf_topics[t]["words"][:5])
        for t in df["nmf_dominant_topic"]
    ]

    df.to_csv(RESULTS_CSV, index=False)
    print(f"    Saved to: {RESULTS_CSV}")

    # [8] Summary table
    _sep()
    print("  TOPIC DISTRIBUTION ACROSS CATEGORIES")
    _sep()
    if "category" in df.columns:
        pivot = (
            df.groupby(["category", "dominant_topic"])
              .size()
              .unstack(fill_value=0)
        )
        pivot.columns = [f"Topic_{c}" for c in pivot.columns]
        print(pivot.to_string())
    _sep()

    # [9] LDA topic summary
    print("  LDA TOPIC SUMMARY (Top 5 words)")
    _sep()
    for t in lda_topics:
        print(f"  Topic {t['topic_id']+1:>2}: {' | '.join(t['words'][:5])}")
    print()
    print("  NMF TOPIC SUMMARY (Top 5 words)")
    _sep("-")
    for t in nmf_topics:
        print(f"  Topic {t['topic_id']+1:>2}: {' | '.join(t['words'][:5])}")
    _sep()

    # [10] JSON summary
    summary_json = {
        "task":            "Task 5 — Topic Modeling",
        "generated":       datetime.now().isoformat(),
        "n_documents":     len(df),
        "n_topics":        lda_model.num_topics,
        "lda_coherence":   round(float(lda_coherence), 4),
        "lda_topics":      [
            {"topic_id": t["topic_id"] + 1, "top_words": t["words"][:N_WORDS]}
            for t in lda_topics
        ],
        "nmf_topics":      [
            {"topic_id": t["topic_id"] + 1, "top_words": t["words"][:N_WORDS]}
            for t in nmf_topics
        ],
        "topic_distribution": df["dominant_topic"].value_counts().to_dict(),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Results saved!")
    print(f"   Per-sample CSV : {RESULTS_CSV}")
    print(f"   Summary JSON   : {SUMMARY_JSON}")
    _sep()


if __name__ == "__main__":
    main()
