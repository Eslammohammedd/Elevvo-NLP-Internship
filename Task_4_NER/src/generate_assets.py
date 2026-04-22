import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def generate_assets():
    assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    # Mock Model Comparison Data for en_core_web_sm vs en_core_web_md
    models = ['en_core_web_sm', 'en_core_web_md']
    entities_extracted = [84, 98]
    accuracy = [0.85, 0.92]

    # Model Performance Results Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=accuracy, palette='Blues_d')
    plt.title('Accuracy Comparison: spaCy SM vs MD Models')
    plt.ylim(0, 1)
    for index, value in enumerate(accuracy):
        plt.text(index, value + 0.02, str(value), ha='center')
    
    plt.savefig(os.path.join(assets_dir, 'model_comparison.png'))
    plt.close()

    # Model Entity Extraction Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=entities_extracted, palette='Greens_d')
    plt.title('Entities Extracted (Sample Dataset): SM vs MD')
    for index, value in enumerate(entities_extracted):
        plt.text(index, value + 1, str(value), ha='center')
    
    plt.savefig(os.path.join(assets_dir, 'entity_extraction_counts.png'))
    plt.close()

    # Create dummy displacy visually
    html_content = """
    <div class="entities" style="line-height: 2.5; direction: ltr">
    <mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Apple Inc.
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
    </mark>
    is considering buying a startup in 
    <mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        London
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
    </mark>
    for 
    <mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        $1 billion
        <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">MONEY</span>
    </mark>
    .
    </div>
    """
    with open(os.path.join(assets_dir, 'displacy_visualization.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    generate_assets()
    print("Assets generated successfully.")
