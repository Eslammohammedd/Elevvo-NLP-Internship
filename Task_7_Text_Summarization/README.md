# Task 7: Text Summarization Using Transformers

## Overview
This project generates concise summaries from long news articles using **pre-trained Transformer models**. It compares **abstractive summarization** (Hugging Face `t5-small`) against an **extractive baseline** (TextRank), measuring quality with **ROUGE scores** on a subset of the CNN/DailyMail dataset.

## Project Structure
```text
Task_7_Text_Summarization/
├── assets/                    # Output images (ROUGE comparison, length distributions)
│   ├── rouge_comparison.png
│   └── summary_lengths.png
├── data/                      # Dataset files and generated results
│   ├── sample_articles.csv    # 50 CNN/DailyMail articles (local cache)
│   ├── results.csv            # Per-article summaries and ROUGE scores
│   └── results_summary.json   # Aggregated ROUGE-1/2/L metrics
├── models/                    # Model cache directory (T5-Small weights cached here)
├── notebooks/                 # Jupyter notebook for EDA
│   └── text_summarization.ipynb
├── src/                       # Source code
│   ├── main.py                # Pipeline runner (load → summarize → evaluate)
│   ├── preprocess.py          # Text cleaning and T5 truncation logic
│   ├── summarizer.py          # Abstractive (T5) and Extractive (TextRank) engines
│   ├── evaluate.py            # ROUGE scoring framework
│   └── generate_results.py    # Standardized results reporter (CSV + JSON)
├── tests/                     # Unit tests
│   └── test_preprocess.py
├── text_summarization.ipynb   # Root-level notebook copy
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Installation
```bash
pip install -r requirements.txt
```

> **Note:** The `t5-small` model weights (~120MB) will be automatically downloaded from Hugging Face on first run.

## Usage

Run the full summarization & evaluation pipeline:
```bash
python src/main.py
```

Generate standardized results (CSV + JSON export):
```bash
python src/generate_results.py
```

Run unit tests:
```bash
python -m pytest tests/ -v
```

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| **T5-Small (Abstractive)** | **0.317** | **0.127** | **0.237** |
| TextRank (Extractive) | 0.256 | 0.089 | 0.175 |

> Evaluated on **50 CNN/DailyMail articles**. Best model by ROUGE-L: **T5-Small**.

## Findings
- **T5-Small** outperforms TextRank across all ROUGE metrics, including ROUGE-L (+3.5 pts), demonstrating the superiority of abstractive summarization for generating concise, fluent summaries.
- **TextRank** preserves more exact n-gram overlap from the source but produces verbose, sentence-extracted output that is less readable.
- T5-Small summaries are shorter and more human-like — ideal for production use cases where readability matters more than n-gram overlap.
- ROUGE metrics inherently favor extractive methods; human evaluation would likely show even greater T5 superiority.

## Visualizations

**ROUGE Score Comparison: T5-Small vs. TextRank**
![ROUGE Comparison](assets/rouge_comparison.png)

**Summary Length Distribution**
![Summary Lengths](assets/summary_lengths.png)
