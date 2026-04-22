import os
import subprocess

def generate_architecture_diagram():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    mermaid_code = """
graph LR
    A(Input Resumes) --> C(Preprocess & Clean)
    B(Input Job Description) --> C
    
    C --> D(NER Extraction - Skills/Experience)
    C --> E(SentenceTransformers)
    
    E --> F(Cosine Similarity Matcher)
    
    D --> G[Ranked Candidates]
    F --> G
    
    style A fill:#E1D5E7
    style B fill:#D5E8D4
    style C fill:#FFE6CC
    style D fill:#FFF2CC
    style E fill:#DAE8FC
    style F fill:#F8CECC
    style G fill:#D5E8D4
"""
    mmd_path = os.path.join(assets_dir, 'architecture.mmd')
    png_path = os.path.join(assets_dir, 'architecture.png')
    
    with open(mmd_path, 'w') as f:
        f.write(mermaid_code)
        
    print("Generating diagram using mermaid-cli...")
    try:
        # Run npx mermaid-cli
        subprocess.run(['npx', '-y', '@mermaid-js/mermaid-cli', '-i', mmd_path, '-o', png_path], check=True, shell=True)
        print(f"Architecture diagram saved to {png_path}")
        
        # Cleanup mmd file
        if os.path.exists(mmd_path):
            os.remove(mmd_path)
    except Exception as e:
        print(f"Error generating diagram: {e}")

if __name__ == "__main__":
    generate_architecture_diagram()
