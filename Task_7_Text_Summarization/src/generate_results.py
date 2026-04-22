"""
generate_results.py — Task 7: Text Summarization
=================================================
Loads the T5-small model (from HuggingFace cache) and TextRank,
runs inference on ALL articles in data/sample_articles.csv,
and saves a comprehensive results report including ROUGE scores.

Output files:
  data/results.csv        → Per-article summaries + ROUGE scores
  data/results_summary.json → Aggregated ROUGE scores + metadata

Usage:
    python src/generate_results.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ── path setup ──────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocess import download_nltk_resources, truncate_for_t5
from summarizer import load_t5_model, generate_t5_summary, textrank_summary
from evaluate   import compute_rouge, print_rouge_scores

# ── config ──────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_CSV  = os.path.join(DATA_DIR, "results.csv")
SUMMARY_JSON = os.path.join(DATA_DIR, "results_summary.json")


# ── helpers ──────────────────────────────────────────────────────────────────
def _sep(char="=", n=60):
    print(char * n)


def _rouge_per_article(pred: str, ref: str) -> dict:
    """Return per-article ROUGE scores as a flat dict."""
    from rouge_score import rouge_scorer as rs_module
    scorer = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    _sep()
    print("  TASK 7 — TEXT SUMMARIZATION  |  Results Reporter")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep()

    # [1] NLTK resources
    print("\n[1] Downloading NLTK resources...")
    download_nltk_resources()

    # [2] Load dataset
    data_path = os.path.join(DATA_DIR, "sample_articles.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "  → Run `python src/main.py` first to generate the dataset."
        )
    print(f"[2] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    N_SAMPLES = len(df)
    print(f"    Articles loaded: {N_SAMPLES}")

    # [3] Load T5 model
    print("\n[3] Loading T5-small model (from cache)...")
    tokenizer, model = load_t5_model("t5-small")
    print("    [OK] T5-small ready")

    # [4] Generate T5 summaries
    print(f"\n[4] Generating T5 summaries for {N_SAMPLES} articles...")
    t5_summaries = []
    for i, row in df.iterrows():
        truncated = truncate_for_t5(row["article"], tokenizer)
        summary   = generate_t5_summary(truncated, tokenizer, model)
        t5_summaries.append(summary)
        if (i + 1) % 10 == 0 or (i + 1) == N_SAMPLES:
            print(f"    → Processed {i+1}/{N_SAMPLES}")
    df["t5_summary"] = t5_summaries

    # [5] Generate TextRank summaries
    print("\n[5] Generating TextRank summaries...")
    df["textrank_summary"] = df["article"].apply(textrank_summary)

    # [6] Per-article ROUGE scores
    print("\n[6] Computing per-article ROUGE scores...")
    refs   = df["reference_summary"].tolist()
    t5_per = [_rouge_per_article(p, r) for p, r in zip(t5_summaries, refs)]
    tr_per = [_rouge_per_article(p, r) for p, r in zip(df["textrank_summary"].tolist(), refs)]

    df["t5_rouge1"] = [s["rouge1"] for s in t5_per]
    df["t5_rouge2"] = [s["rouge2"] for s in t5_per]
    df["t5_rougeL"] = [s["rougeL"] for s in t5_per]
    df["tr_rouge1"] = [s["rouge1"] for s in tr_per]
    df["tr_rouge2"] = [s["rouge2"] for s in tr_per]
    df["tr_rougeL"] = [s["rougeL"] for s in tr_per]

    df.to_csv(RESULTS_CSV, index=False)
    print(f"    Saved to: {RESULTS_CSV}")

    # [7] Aggregate ROUGE scores
    t5_rouge       = compute_rouge(df["t5_summary"].tolist(), refs)
    textrank_rouge = compute_rouge(df["textrank_summary"].tolist(), refs)

    _sep()
    print("  AGGREGATE ROUGE SCORES")
    _sep()
    print_rouge_scores(t5_rouge,       "T5-Small (Abstractive)")
    print_rouge_scores(textrank_rouge, "TextRank (Extractive)")

    # [8] Comparison table
    comparison = pd.DataFrame({
        "Metric":   ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "T5-Small": [round(t5_rouge["ROUGE-1"], 4),
                     round(t5_rouge["ROUGE-2"], 4),
                     round(t5_rouge["ROUGE-L"], 4)],
        "TextRank": [round(textrank_rouge["ROUGE-1"], 4),
                     round(textrank_rouge["ROUGE-2"], 4),
                     round(textrank_rouge["ROUGE-L"], 4)],
    })
    comparison["Winner"] = comparison.apply(
        lambda r: "T5-Small" if r["T5-Small"] >= r["TextRank"] else "TextRank", axis=1
    )
    _sep()
    print("  MODEL COMPARISON TABLE")
    _sep()
    print(comparison.to_string(index=False))
    _sep()

    best_model = (
        "T5-Small"
        if t5_rouge["ROUGE-L"] >= textrank_rouge["ROUGE-L"]
        else "TextRank"
    )
    print(f"\n  [BEST] Best Model (ROUGE-L): {best_model}")

    # [9] Sample output showcase
    _sep()
    print("  SAMPLE OUTPUTS (First Article)")
    _sep()
    row0 = df.iloc[0]
    print(f"  Article (first 300 chars):\n  {row0['article'][:300]}...\n")
    print(f"  Reference : {row0['reference_summary']}\n")
    print(f"  T5-Small  : {row0['t5_summary']}\n")
    print(f"  TextRank  : {row0['textrank_summary']}\n")

    # [10] JSON summary
    summary_json = {
        "task":       "Task 7 — Text Summarization",
        "generated":  datetime.now().isoformat(),
        "n_articles": N_SAMPLES,
        "models": {
            "T5-Small (Abstractive)": {
                "ROUGE-1": round(t5_rouge["ROUGE-1"], 4),
                "ROUGE-2": round(t5_rouge["ROUGE-2"], 4),
                "ROUGE-L": round(t5_rouge["ROUGE-L"], 4),
            },
            "TextRank (Extractive)": {
                "ROUGE-1": round(textrank_rouge["ROUGE-1"], 4),
                "ROUGE-2": round(textrank_rouge["ROUGE-2"], 4),
                "ROUGE-L": round(textrank_rouge["ROUGE-L"], 4),
            },
        },
        "best_model_rouge_L": best_model,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Results saved!")
    print(f"   Per-article CSV : {RESULTS_CSV}")
    print(f"   Summary JSON    : {SUMMARY_JSON}")
    _sep()


if __name__ == "__main__":
    main()
