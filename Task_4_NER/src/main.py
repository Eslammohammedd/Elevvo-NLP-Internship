import os
import pandas as pd
from typing import List
from ner import load_models, extract_entities_model, extract_entities_rule_based
import sys

def main():
    print("Starting Task 4: Named Entity Recognition pipeline...")

    # For demonstration, we'll use a sample paragraph since downloading full CoNLL003 can be tricky.
    # In a real scenario, we'd load the dataset from 'data/train.csv'.
    sample_texts = [
        "Apple Inc. is considering buying a startup in London for $1 billion.",
        "Elon Musk announced that Tesla will open a new Gigafactory in Texas next year.",
        "The United Nations held a meeting in New York City regarding climate change."
    ]

    print("\nLoading spaCy models (en_core_web_sm and en_core_web_md)...")
    nlp_sm, nlp_md = load_models()

    for idx, text in enumerate(sample_texts):
        print(f"\n--- Document {idx + 1} ---")
        print(f"Text: {text}")
        
        # 1. Rule-based extraction
        rule_based_ents = extract_entities_rule_based(text)
        print("\nRule-Based Entities:")
        for ent in rule_based_ents:
            print(f" - {ent['text']} ({ent['label']})")
        
        # 2. Model-based (Small model)
        sm_ents = extract_entities_model(text, nlp_sm)
        print("\nModel-Based Entities (en_core_web_sm):")
        for ent in sm_ents:
            print(f" - {ent['text']} ({ent['label']})")
            
        # 3. Model-based (Medium model)
        md_ents = extract_entities_model(text, nlp_md)
        print("\nModel-Based Entities (en_core_web_md):")
        for ent in md_ents:
            print(f" - {ent['text']} ({ent['label']})")

    # Bonus: Generate displaCy HTML for one example
    try:
        from spacy import displacy
        
        doc = nlp_md(sample_texts[0])
        html = displacy.render(doc, style="ent", page=True)
        
        assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
        os.makedirs(assets_dir, exist_ok=True)
        
        output_path = os.path.join(assets_dir, 'displacy_visualization.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"\nSaved displaCy visualization to {output_path}")
    except Exception as e:
        print(f"\nFailed to generate displaCy visualization: {e}")

if __name__ == "__main__":
    main()
