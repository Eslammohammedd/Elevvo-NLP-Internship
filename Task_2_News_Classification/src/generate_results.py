"""
generate_results.py — Task 2: News Classification
==================================================
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
from train import LABEL_MAP   # {0:'World', 1:'Sports', 2:'Business', 3:'Sci/Tech'}

# ── config ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_CSV  = os.path.join(DATA_DIR, "results.csv")
SUMMARY_JSON = os.path.join(DATA_DIR, "results_summary.json")

CATEGORY_NAMES = list(LABEL_MAP.values())   # ['World','Sports','Business','Sci/Tech']


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
    return {
        "model":          model_name,
        "accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "macro_precision":round(precision_score(y_true, y_pred, average="macro",  zero_division=0), 4),
        "macro_recall":   round(recall_score(y_true, y_pred,    average="macro",  zero_division=0), 4),
        "macro_f1":       round(f1_score(y_true, y_pred,        average="macro",  zero_division=0), 4),
        "weighted_f1":    round(f1_score(y_true, y_pred,        average="weighted",zero_division=0), 4),
    }


# ── inference ────────────────────────────────────────────────────────────────
def run_inference(clf, X_test, model_name: str, y_true, label_names):
    y_pred = clf.predict(X_test)
    print(f"\n  [{model_name}]")
    print(
        classification_report(y_true, y_pred, target_names=label_names)
    )
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True:{n}" for n in label_names],
        columns=[f"Pred:{n}" for n in label_names],
    )
    print("  Confusion Matrix:")
    print(cm_df.to_string())
    return y_pred, _metrics_dict(y_true, y_pred, model_name)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    _sep()
    print("  TASK 2 — NEWS CLASSIFICATION  |  Results Reporter")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep()

    # [1] resources
    print("\n[1] Downloading NLTK resources...")
    download_nltk_resources()

    # [2] load data
    print("[2] Loading test dataset...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_df["category"] = test_df["label"].map(LABEL_MAP)
    print(f"    Test samples : {len(test_df):,}")
    print(f"    Class balance:\n{test_df['category'].value_counts().to_string()}")

    # [3] preprocess
    print("\n[3] Preprocessing test set...")
    test_df["clean_text"] = test_df["text"].apply(preprocess_text)
    y_true = test_df["label"].values

    # [4] load artifacts
    print("[4] Loading vectorizer & models from models/...")
    tfidf = _load_model("tfidf_vectorizer.pkl")
    X_test = tfidf.transform(test_df["clean_text"])

    model_files = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Linear SVM":          "linear_svm_model.pkl",
        "Random Forest":       "random_forest_model.pkl",
    }

    _sep()
    print("  MODEL COMPARISON")
    _sep()

    all_preds    = {}
    summary_rows = []
    for name, fname in model_files.items():
        clf = _load_model(fname)
        y_pred, metrics = run_inference(
            clf, X_test, name, y_true, CATEGORY_NAMES
        )
        all_preds[name]    = y_pred
        summary_rows.append(metrics)
        _sep("-")

    # [5] per-sample results CSV
    print("[5] Saving per-sample results...")
    results_df = test_df[["text", "label", "category"]].copy()
    results_df.rename(columns={"label": "true_label", "category": "true_category"}, inplace=True)
    for name, y_pred in all_preds.items():
        safe = name.replace(" ", "_").lower()
        results_df[f"{safe}_pred"]     = y_pred
        results_df[f"{safe}_category"] = pd.Series(y_pred).map(LABEL_MAP).values
        results_df[f"{safe}_correct"]  = (y_pred == y_true)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"    Saved to: {RESULTS_CSV}")

    # [6] summary table
    _sep()
    print("  OVERALL SUMMARY TABLE")
    _sep()
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    best = max(summary_rows, key=lambda r: r["accuracy"])
    print(f"\n  [BEST] Best Model: {best['model']}  (Accuracy: {best['accuracy']:.4f})")
    _sep()

    # [7] save JSON summary
    summary_json = {
        "task":         "Task 2 — News Classification",
        "generated":    datetime.now().isoformat(),
        "test_samples": int(len(y_true)),
        "categories":   CATEGORY_NAMES,
        "models":       summary_rows,
        "best_model":   best["model"],
        "best_accuracy":best["accuracy"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Results saved!")
    print(f"   Per-sample CSV : {RESULTS_CSV}")
    print(f"   Summary JSON   : {SUMMARY_JSON}")
    _sep()


if __name__ == "__main__":
    main()
