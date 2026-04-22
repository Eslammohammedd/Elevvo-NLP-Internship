"""
generate_results.py
Evaluates QA models on the dataset and saves detailed results.
"""
import os
import sys
import json
import ast
import pandas as pd
from datetime import datetime

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from qa_model import load_qa_pipeline, answer_question
from evaluate import compute_exact_match, compute_f1, compute_metrics

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
SUMMARY_JSON = os.path.join(DATA_DIR, "results_summary.json")

def _sep(char="=", n=60):
    print(char * n)

def main():
    _sep()
    print("  TASK 6 — QUESTION ANSWERING  |  Results Reporter")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _sep()

    data_path = os.path.join(DATA_DIR, "sample_squad.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"[1] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    N_SAMPLES = len(df)
    print(f"    Questions loaded: {N_SAMPLES}")

    print("\n[2] Loading QA model (distilbert-base-cased-distilled-squad)...")
    qa_pipeline = load_qa_pipeline()
    print("    [OK] Model ready")

    print(f"\n[3] Generating answers for {N_SAMPLES} questions...")
    predictions = []
    truths = []
    scores = []
    em_scores = []
    f1_scores = []

    for i, row in df.iterrows():
        context = row['context']
        question = row['question']
        
        try:
            answers_dict = ast.literal_eval(row['answers'])
            truth = answers_dict['text'][0] if len(answers_dict['text']) > 0 else ""
        except:
            truth = str(row['answers'])

        result = answer_question(qa_pipeline, context, question)
        pred = result['answer']
        
        predictions.append(pred)
        truths.append(truth)
        scores.append(round(result['score'], 4))
        
        em_scores.append(compute_exact_match(pred, truth))
        f1_scores.append(round(compute_f1(pred, truth), 4))
        
        if (i + 1) % 20 == 0 or (i + 1) == N_SAMPLES:
            print(f"    → Processed {i+1}/{N_SAMPLES}")

    df["reference_answer"] = truths
    df["predicted_answer"] = predictions
    df["confidence_score"] = scores
    df["exact_match"] = em_scores
    df["f1_score"] = f1_scores

    # Save CSV
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[4] Saved detailed results to: {RESULTS_CSV}")

    # Aggregate scores
    metrics = compute_metrics(predictions, truths)
    
    _sep()
    print("  AGGREGATE METRICS")
    _sep()
    print(f"  Exact Match (EM): {metrics['exact_match']}%")
    print(f"  F1 Score:         {metrics['f1']}%")
    _sep()

    # JSON summary
    summary_json = {
        "task": "Task 6 — Question Answering",
        "generated": datetime.now().isoformat(),
        "n_questions": N_SAMPLES,
        "model": "distilbert-base-cased-distilled-squad",
        "metrics": {
            "Exact Match (%)": metrics["exact_match"],
            "F1 Score (%)": metrics["f1"]
        }
    }
    
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
        
    print(f"   Summary JSON saved to: {SUMMARY_JSON}")
    _sep()

if __name__ == "__main__":
    main()
