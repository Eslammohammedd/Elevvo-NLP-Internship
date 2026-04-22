import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Task 6: Question Answering Exploration\nIn this notebook, we explore the SQuAD dataset and test our QA models interactively."),
    nbf.v4.new_code_cell("import pandas as pd\nimport sys\nimport os\n\nsys.path.insert(0, os.path.abspath('../src'))\nfrom qa_model import load_qa_pipeline, answer_question\nfrom evaluate import compute_exact_match, compute_f1"),
    nbf.v4.new_markdown_cell("## 1. Load Data"),
    nbf.v4.new_code_cell("df = pd.read_csv('../data/sample_squad.csv')\ndf.head()"),
    nbf.v4.new_markdown_cell("## 2. Load Model"),
    nbf.v4.new_code_cell("qa_pipeline = load_qa_pipeline()"),
    nbf.v4.new_markdown_cell("## 3. Interactive Testing"),
    nbf.v4.new_code_cell("context = df.iloc[0]['context']\nquestion = df.iloc[0]['question']\n\nprint('Context:', context)\nprint('\\nQuestion:', question)\n\nresult = answer_question(qa_pipeline, context, question)\nprint('\\nPredicted Answer:', result['answer'])\nprint('Confidence:', result['score'])")
]

nb['cells'] = cells

os.makedirs('Task_6_Question_Answering/notebooks', exist_ok=True)
with open('Task_6_Question_Answering/notebooks/question_answering.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
