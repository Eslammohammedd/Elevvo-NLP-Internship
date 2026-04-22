"""
generate_results.py — Task 1: Sentiment Analysis
=================================================
Loads the trained models from models/ and runs inference on the test set.
Saves a detailed, structured results report to data/results.csv.

Usage:
    python src/generate_results.py
"""

import os
import sys
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ── path setup ──────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocess import download_nltk_resources, preprocess_text


# ── config ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
SUMMARY_JSON = os.path.join(DATA_DIR, "results_summary.json")

LABEL_NAMES = {0: "Negative", 1: "Positive"}


# ── helpers ──────────────────────────────────────────────────────────────────
def _sep(char="=", n=60):
    print(char * n)


def _load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "  → Run `python src/main.py` first to train and save the models."
        )
    return joblib.load(path)


def _metrics_dict(y_true, y_pred, model_name: str) -> dict:
    """Return a flat metrics dict for one model."""
    return {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    _sep()
    print("  TASK 1 — SENTIMENT ANALYSIS  |  Results Reporter")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep()

    # [1] resources
    print("\n[1] Downloading NLTK resources...")
    download_nltk_resources()

    # [2] load data
    print("[2] Loading test dataset...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"    Test samples : {len(test_df):,}")
    print(f"    Class balance: {test_df['label'].value_counts().to_dict()}")

    # [3] preprocess
    print("[3] Preprocessing test set...")
    test_df["clean_text"] = test_df["text"].apply(preprocess_text)

    y_true = test_df["label"].values

    # [4] load vectorizers
    print("[4] Loading vectorizers & models from models/...")
    tfidf = _load_model("tfidf_vectorizer.pkl")
    cv    = _load_model("count_vectorizer.pkl")
    lr    = _load_model("logistic_regression_model.pkl")
    nb    = _load_model("naive_bayes_model.pkl")

    X_test_tfidf = tfidf.transform(test_df["clean_text"])
    X_test_cv    = cv.transform(test_df["clean_text"])

    # [5] inference
    print("[5] Running inference...")
    y_pred_lr = lr.predict(X_test_tfidf)
    y_pred_nb = nb.predict(X_test_cv)

    # [6] per-sample results CSV
    print("[6] Building per-sample results dataframe...")
    results_df = test_df[["text", "label"]].copy()
    results_df["true_label_name"]    = results_df["label"].map(LABEL_NAMES)
    results_df["lr_prediction"]      = y_pred_lr
    results_df["lr_label_name"]      = pd.Series(y_pred_lr).map(LABEL_NAMES).values
    results_df["lr_correct"]         = (y_pred_lr == y_true)
    results_df["nb_prediction"]      = y_pred_nb
    results_df["nb_label_name"]      = pd.Series(y_pred_nb).map(LABEL_NAMES).values
    results_df["nb_correct"]         = (y_pred_nb == y_true)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"    Saved to: {RESULTS_CSV}")

    # [7] summary metrics
    _sep()
    print("  MODEL COMPARISON SUMMARY")
    _sep()

    lr_metrics = _metrics_dict(y_true, y_pred_lr, "Logistic Regression (TF-IDF)")
    nb_metrics = _metrics_dict(y_true, y_pred_nb, "Naive Bayes (CountVec)")

    summary_rows = [lr_metrics, nb_metrics]
    summary_df   = pd.DataFrame(summary_rows)

    print(summary_df.to_string(index=False))
    _sep()

    for metrics, y_pred, name in [
        (lr_metrics, y_pred_lr, "Logistic Regression (TF-IDF)"),
        (nb_metrics, y_pred_nb, "Naive Bayes (CountVec)"),
    ]:
        print(f"\n  Classification Report — {name}")
        print(
            classification_report(
                y_true, y_pred,
                target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            )
        )
        cm = confusion_matrix(y_true, y_pred)
        print("  Confusion Matrix (rows=True, cols=Predicted):")
        cm_df = pd.DataFrame(
            cm,
            index=[f"True:{LABEL_NAMES[i]}" for i in range(2)],
            columns=[f"Pred:{LABEL_NAMES[i]}" for i in range(2)],
        )
        print(cm_df.to_string())
        _sep("-")

    # [8] save JSON summary
    best = max(summary_rows, key=lambda r: r["accuracy"])
    summary_json = {
        "task":        "Task 1 — Sentiment Analysis",
        "generated":   datetime.now().isoformat(),
        "test_samples": int(len(y_true)),
        "models":      summary_rows,
        "best_model":  best["model"],
        "best_accuracy": best["accuracy"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Results saved!")
    print(f"   Per-sample CSV : {RESULTS_CSV}")
    print(f"   Summary JSON   : {SUMMARY_JSON}")
    _sep()


if __name__ == "__main__":
    main()
