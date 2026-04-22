import os
import pandas as pd

from preprocess import download_nltk_resources, truncate_for_t5
from summarizer import load_t5_model, generate_t5_summary, textrank_summary
from evaluate import compute_rouge, print_rouge_scores

def main():
    print("Downloading NLTK resources...")
    download_nltk_resources()
    
    print("Loading CNN/DailyMail dataset from data/ directory...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, 'data', 'sample_articles.csv'))
    N_SAMPLES = len(df)
    
    print("Loading T5-Small model...")
    tokenizer, model = load_t5_model('t5-small')
    
    print("Truncating articles for T5 input...")
    df['truncated_article'] = df['article'].apply(lambda x: truncate_for_t5(x, tokenizer))
    
    print(f"Generating T5 summaries for {N_SAMPLES} articles...")
    t5_summaries = []
    for i, row in df.iterrows():
        summary = generate_t5_summary(row['truncated_article'], tokenizer, model)
        t5_summaries.append(summary)
        if (i + 1) % 10 == 0:
            print(f"  -> Processed {i+1}/{N_SAMPLES} articles")
    df['t5_summary'] = t5_summaries
    
    print("Generating TextRank summaries...")
    df['textrank_summary'] = df['article'].apply(textrank_summary)
    
    print("Evaluating ROUGE scores...")
    refs = df['reference_summary'].tolist()
    t5_rouge = compute_rouge(df['t5_summary'].tolist(), refs)
    textrank_rouge = compute_rouge(df['textrank_summary'].tolist(), refs)
    
    print_rouge_scores(t5_rouge, "T5-Small (Abstractive)")
    print_rouge_scores(textrank_rouge, "TextRank (Extractive)")
    
    print("\n✅ Pipeline complete!")

if __name__ == "__main__":
    main()
