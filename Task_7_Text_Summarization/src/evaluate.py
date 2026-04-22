import numpy as np
from rouge_score import rouge_scorer

def compute_rouge(predictions, references):
    """Compute average ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_scores, r2_scores, rL_scores = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'ROUGE-1': np.mean(r1_scores),
        'ROUGE-2': np.mean(r2_scores),
        'ROUGE-L': np.mean(rL_scores),
    }

def print_rouge_scores(rouge_dict, model_name):
    print('=' * 50)
    print(f'ROUGE SCORES — {model_name}')
    print('=' * 50)
    for k, v in rouge_dict.items():
        print(f'  {k}: {v:.4f}')
