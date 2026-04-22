"""
main.py
Main pipeline runner for Task 6.
Loads the sample SQuAD data, runs the QA model on it, and prints the results.
"""
import os
import ast
import pandas as pd
from qa_model import load_qa_pipeline, answer_question
from evaluate import compute_metrics

def main():
    print("="*60)
    print("  TASK 6: QUESTION ANSWERING PIPELINE")
    print("="*60)

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_squad.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        print("Please run download_data.py in the root directory first.")
        return
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")

    # 2. Load Model
    print("\nLoading QA model...")
    qa_pipeline = load_qa_pipeline()

    # 3. Run Inference
    print("\nRunning inference on sample dataset...")
    predictions = []
    truths = []

    for idx, row in df.iterrows():
        context = row['context']
        question = row['question']
        
        # SQuAD answers column is a stringified dictionary like "{'text': ['answer'], 'answer_start': [123]}"
        try:
            answers_dict = ast.literal_eval(row['answers'])
            if len(answers_dict['text']) > 0:
                truth = answers_dict['text'][0]
            else:
                truth = ""
        except:
            truth = str(row['answers'])

        result = answer_question(qa_pipeline, context, question)
        
        predictions.append(result['answer'])
        truths.append(truth)
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    # 4. Evaluate
    print("\nEvaluating results...")
    metrics = compute_metrics(predictions, truths)
    
    print("="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  Exact Match (EM): {metrics['exact_match']}%")
    print(f"  F1 Score:         {metrics['f1']}%")
    print("="*60)

if __name__ == "__main__":
    main()
