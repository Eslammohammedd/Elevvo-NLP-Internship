"""
test_qa.py
Unit tests for the QA pipeline.
"""
import sys
import os
import pytest

# Add src to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from qa_model import load_qa_pipeline, answer_question
from evaluate import compute_exact_match, compute_f1

@pytest.fixture(scope="module")
def qa_pipeline():
    # We use a very small model for testing if possible, but distilbert is fine
    return load_qa_pipeline()

def test_model_inference(qa_pipeline):
    context = "Elevvo is a leading tech company located in Egypt."
    question = "Where is Elevvo located?"
    
    result = answer_question(qa_pipeline, context, question)
    
    assert "answer" in result
    assert "score" in result
    assert result["answer"].lower() == "egypt"

def test_evaluate_exact_match():
    assert compute_exact_match("Egypt", "Egypt") == 1
    assert compute_exact_match("egypt", "Egypt") == 1
    assert compute_exact_match("The Egypt", "Egypt") == 1
    assert compute_exact_match("Cairo", "Egypt") == 0

def test_evaluate_f1():
    assert compute_f1("Egypt", "Egypt") == 1.0
    assert compute_f1("Cairo", "Egypt") == 0.0
    
    # Partial match
    # pred: "Cairo, Egypt", truth: "Egypt"
    # f1 = 0.666...
    f1 = compute_f1("Cairo, Egypt", "Egypt")
    assert round(f1, 2) == 0.67
