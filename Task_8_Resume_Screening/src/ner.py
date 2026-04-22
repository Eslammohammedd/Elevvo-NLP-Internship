import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# A basic list of skills to match against if we want rule-based skill extraction
COMMON_SKILLS = {
    "python", "java", "c++", "sql", "pandas", "scikit-learn", "machine learning",
    "deep learning", "nlp", "html", "css", "javascript", "react", "node.js",
    "postgresql", "aws", "docker", "kubernetes", "pytorch", "tensorflow", "spacy"
}

def extract_entities(text: str) -> dict:
    """
    Extracts skills and experience (organizations, dates) from the resume text.
    """
    doc = nlp(text)
    
    extracted = {
        "skills": set(),
        "experience": set(),
        "education": set()
    }
    
    # Extract based on NER tags
    for ent in doc.ents:
        if ent.label_ in ["ORG"]:
            extracted["experience"].add(ent.text)
        elif ent.label_ in ["DATE"]:
            extracted["experience"].add(ent.text)
            
    # Extract skills based on exact match from common skills list
    # and also look at noun chunks
    lower_text = text.lower()
    for skill in COMMON_SKILLS:
        if skill in lower_text:
            extracted["skills"].add(skill.capitalize())
            
    # Convert sets to lists for JSON serialization later
    return {
        "skills": list(extracted["skills"]),
        "experience": list(extracted["experience"]),
        "education": list(extracted["education"])
    }
