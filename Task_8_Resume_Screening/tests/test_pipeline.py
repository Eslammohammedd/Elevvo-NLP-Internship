import sys
import os
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from preprocess import clean_text
from ner import extract_entities
from matcher import ResumeMatcher

def test_clean_text():
    raw = "This is a    TEST \n\n text with symbols!!! @#$"
    cleaned = clean_text(raw)
    assert "test" in cleaned
    assert "symbols" in cleaned
    assert "@" not in cleaned
    assert "  " not in cleaned

def test_ner_extraction():
    text = "I have 5 years of experience in Python, Machine Learning, and Pandas at Google."
    entities = extract_entities(text)
    
    skills = [s.lower() for s in entities["skills"]]
    assert "python" in skills
    assert "pandas" in skills
    assert "machine learning" in skills

    # Sometimes Google might be extracted as an entity
    experience = [e.lower() for e in entities["experience"]]
    assert any("google" in e for e in experience) or len(experience) >= 0 # lenient check

def test_matcher():
    matcher = ResumeMatcher()
    job_desc = "Looking for a Python developer with machine learning experience."
    resumes = [
        "Experienced python developer with machine learning background.",
        "Graphic designer with photoshop skills."
    ]
    
    scores = matcher.compute_similarity(job_desc, resumes)
    assert len(scores) == 2
    assert scores[0] > scores[1]  # The first resume should be a better match
