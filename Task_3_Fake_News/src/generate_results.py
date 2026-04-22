"""
generate_results.py — Task 3: Fake News Detection
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

from preprocess import download_nltk_resources, preprocess_text, to_binary

# ── config ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_CSV  = os.path.join(DATA_DIR, "results.csv")
SUMMARY_JSON = os.path.join(DATA_DIR, "results_summary.json")

LABEL_NAMES = {0: "Fake", 1: "Real"}


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
        "precision_fake": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "precision_real": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_fake":    round(recall_score(y_true, y_pred,    pos_label=0, zero_division=0), 4),
        "recall_real":    round(recall_score(y_true, y_pred,    pos_label=1, zero_division=0), 4),
        "f1_weighted":    round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_macro":       round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    _sep()
    print("  TASK 3 — FAKE NEWS DETECTION  |  Results Reporter")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep()

    # [1] resources
    print("\n[1] Downloading NLTK resources...")
    download_nltk_resources()

    # [2] load data
    print("[2] Loading test dataset...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_df["binary_label"] = test_df["label"].apply(to_binary)
    test_df["label_name"]   = test_df["binary_label"].map(LABEL_NAMES)
    print(f"    Test samples  : {len(test_df):,}")
    print(f"    Label distribution:\n{test_df['label_name'].value_counts().to_string()}")

    # [3] preprocess
    print("\n[3] Preprocessing statements...")
    test_df["clean_statement"] = test_df["statement"].apply(preprocess_text)
    y_true = test_df["binary_label"].values

    # [4] load artifacts
    print("[4] Loading vectorizer & models from models/...")
    tfidf  = _load_model("tfidf_vectorizer.pkl")
    lr_clf = _load_model("logistic_regression_model.pkl")
    svm_clf = _load_model("linear_svm_model.pkl")
    X_test = tfidf.transform(test_df["clean_statement"])

    # [5] inference
    print("\n[5] Running inference...")
    model_map = {
        "Logistic Regression": lr_clf,
        "Linear SVM":          svm_clf,
    }

    _sep()
    print("  DETAILED RESULTS PER MODEL")
    _sep()

    all_preds    = {}
    summary_rows = []

    for name, clf in model_map.items():
        y_pred = clf.predict(X_test)
        all_preds[name] = y_pred
        metrics = _metrics_dict(y_true, y_pred, name)
        summary_rows.append(metrics)

        print(f"\n  ── {name} ──")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1 Macro  : {metrics['f1_macro']:.4f}")
        print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
        print()
        print(
            classification_report(
                y_true, y_pred,
                target_names=["Fake", "Real"],
            )
        )
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["True:Fake", "True:Real"],
            columns=["Pred:Fake", "Pred:Real"],
        )
        print("  Confusion Matrix:")
        print(cm_df.to_string())
        _sep("-")

    # [6] per-sample results CSV
    print("\n[6] Saving per-sample results...")
    results_df = test_df[["statement", "binary_label", "label_name"]].copy()
    results_df.rename(
        columns={"binary_label": "true_label", "label_name": "true_label_name"}, inplace=True
    )
    for name, y_pred in all_preds.items():
        safe = name.replace(" ", "_").lower()
        results_df[f"{safe}_pred"]       = y_pred
        results_df[f"{safe}_label_name"] = pd.Series(y_pred).map(LABEL_NAMES).values
        results_df[f"{safe}_correct"]    = (y_pred == y_true)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"    Saved to: {RESULTS_CSV}")

    # [7] summary table
    _sep()
    print("  OVERALL SUMMARY TABLE")
    _sep()
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    best = max(summary_rows, key=lambda r: r["f1_macro"])
    print(f"\n  [BEST] Best Model (F1 Macro): {best['model']}  ({best['f1_macro']:.4f})")
    _sep()

    # [8] JSON summary
    summary_json = {
        "task":         "Task 3 — Fake News Detection",
        "generated":    datetime.now().isoformat(),
        "test_samples": int(len(y_true)),
        "classes":      list(LABEL_NAMES.values()),
        "models":       summary_rows,
        "best_model":   best["model"],
        "best_f1_macro":best["f1_macro"],
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Results saved!")
    print(f"   Per-sample CSV : {RESULTS_CSV}")
    print(f"   Summary JSON   : {SUMMARY_JSON}")
    _sep()


if __name__ == "__main__":
    main()
