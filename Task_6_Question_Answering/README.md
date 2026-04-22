# Task 6: Question Answering with Transformers

This repository contains the implementation of **Task 6: Question Answering with Transformers** as part of the Elevvo NLP Internship. The goal is to build a system that can answer questions based on a given context or passage using pre-trained transformer models fine-tuned for question answering.

## 🚀 Features
- **Transformer-based NLP**: Uses Hugging Face pre-trained models (`distilbert-base-cased-distilled-squad`) to extract answer spans.
- **Span Extraction**: Feeds the model a context and a question, extracting the precise answer text and its confidence score.
- **Evaluation Metrics**: Evaluates performance using Exact Match (EM) and F1 Score metrics.
- **Interactive UI**: Includes a Streamlit web application for interactive testing (Bonus).

## 📂 Project Structure
```text
Task_6_Question_Answering/
├── data/
│   ├── sample_squad.csv        # 100 sample records from SQuAD v1.1
│   ├── results.csv             # Evaluation results per question
│   └── results_summary.json    # Aggregated metrics (EM & F1)
├── src/
│   ├── main.py                 # Unified pipeline runner
│   ├── qa_model.py             # Model loading and answer extraction logic
│   ├── evaluate.py             # Evaluation metrics calculation (EM and F1)
│   └── generate_results.py     # Script to evaluate and save standardized results
├── tests/
│   └── test_qa.py              # Unit tests for the QA pipeline
├── notebooks/
│   └── question_answering.ipynb# Jupyter notebook for exploration and testing
├── streamlit_app.py            # Streamlit interactive web UI
├── requirements.txt            # Project dependencies
└── README.md                   # This documentation file
```

## ⚙️ Installation & Setup
1. **Navigate to the project directory:**
   ```bash
   cd Task_6_Question_Answering
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage
1. **Run the Main Pipeline:**
   Executes the QA model on the sample dataset.
   ```bash
   python src/main.py
   ```
2. **Generate Standardized Results:**
   Evaluates the model, calculating F1 and Exact Match scores, and saves them to the `data/` directory.
   ```bash
   python src/generate_results.py
   ```
3. **Run Unit Tests:**
   ```bash
   pytest tests/
   ```
4. **Launch the Interactive Web UI:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Evaluation
The model is evaluated based on:
- **Exact Match (EM):** Percentage of predictions that match any one of the ground truth answers exactly.
- **F1 Score:** Measures the average overlap between the prediction and ground truth answers.
