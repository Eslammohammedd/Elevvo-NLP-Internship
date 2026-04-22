"""
evaluate.py
Functions for evaluating Question Answering models (Exact Match and F1 score).
"""
import re
import string
import collections

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        if type(text) == str:
            return text.lower()
        return text
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact_match(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))

def compute_f1(prediction, truth):
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(truth)
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(predictions, truths):
    """
    Compute average exact match and f1 score over a list of predictions and truths.
    """
    assert len(predictions) == len(truths)
    em_total = 0
    f1_total = 0
    for pred, truth in zip(predictions, truths):
        em_total += compute_exact_match(pred, truth)
        f1_total += compute_f1(pred, truth)
    
    avg_em = em_total / len(predictions)
    avg_f1 = f1_total / len(predictions)
    
    return {"exact_match": round(avg_em * 100, 2), "f1": round(avg_f1 * 100, 2)}
