import spacy
from typing import List, Dict, Tuple
import re

def load_models() -> Tuple[spacy.language.Language, spacy.language.Language]:
    # Load small and medium models to compare
    try:
        nlp_sm = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp_sm = spacy.load("en_core_web_sm")

    try:
        nlp_md = spacy.load("en_core_web_md")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
        nlp_md = spacy.load("en_core_web_md")

    return nlp_sm, nlp_md

def extract_entities_model(text: str, nlp_model: spacy.language.Language) -> List[Dict[str, str]]:
    """Extract named entities using a spaCy model."""
    doc = nlp_model(text)
    entities = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
    return entities

def extract_entities_rule_based(text: str) -> List[Dict[str, str]]:
    """A simple rule-based approach to extract capitalized words which might be Proper Nouns/Entities."""
    # A simple regex that matches sequences of capitalized words
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    matches = re.finditer(pattern, text)
    
    entities = []
    # Filter out common stop words that might be capitalized at the beginning of a sentence
    stop_words = {"The", "A", "An", "In", "On", "Of", "And", "To", "With", "For", "By", "At"}
    for match in matches:
        entity_text = match.group(1)
        if entity_text not in stop_words:
            entities.append({
                "text": entity_text,
                "label": "RULE_BASED_ENTITY",
                "start": match.start(),
                "end": match.end()
            })
    return entities
