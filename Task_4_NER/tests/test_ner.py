import pytest
import spacy
from src.ner import extract_entities_rule_based, extract_entities_model

@pytest.fixture
def nlp_sm():
    return spacy.load("en_core_web_sm")

def test_extract_entities_rule_based():
    text = "Apple is buying a startup in London."
    entities = extract_entities_rule_based(text)
    
    texts = [e["text"] for e in entities]
    assert "Apple" in texts
    assert "London" in texts

def test_extract_entities_model(nlp_sm):
    text = "Apple format is buying a startup in London."
    entities = extract_entities_model(text, nlp_sm)
    
    labels = [e["label"] for e in entities]
    texts = [e["text"] for e in entities]
    
    assert "Apple" in texts
    assert "London" in texts
    
    assert "ORG" in labels or "GPE" in labels
