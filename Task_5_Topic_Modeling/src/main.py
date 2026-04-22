import os
import sys
import pandas as pd

# Ensure src/ modules are importable when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import download_nltk_resources, preprocess_corpus
from topic_model import TopicModelEngine
from evaluate import compute_coherence, print_topics, print_coherence

N_TOPICS   = 5
N_WORDS    = 10
LDA_PASSES = 15

def main():
    print("\n" + "=" * 50)
    print("  Topic Modeling on News Articles")
    print("=" * 50)

    # ------------------------------------------------------------------
    # [1] Download resources
    # ------------------------------------------------------------------
    print("\n[1] Downloading NLTK resources...")
    download_nltk_resources()

    # ------------------------------------------------------------------
    # [2] Load dataset
    # ------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'sample_news.csv')
    print(f"\n[2] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"    Found {len(df)} articles across {df['category'].nunique()} categories.")

    raw_texts = df['text'].tolist()

    # ------------------------------------------------------------------
    # [3] Preprocess
    # ------------------------------------------------------------------
    print("\n[3] Preprocessing text (tokenization, stopword removal, lemmatization)...")
    tokenized_docs = preprocess_corpus(raw_texts)
    avg_tokens = sum(len(d) for d in tokenized_docs) / len(tokenized_docs)
    print(f"    Average tokens per document after cleaning: {avg_tokens:.1f}")

    # ------------------------------------------------------------------
    # [4] LDA Model
    # ------------------------------------------------------------------
    print(f"\n[4] Training LDA model (n_topics={N_TOPICS}, passes={LDA_PASSES})...")
    engine = TopicModelEngine(n_topics=N_TOPICS)
    engine.build_lda(tokenized_docs, passes=LDA_PASSES)

    lda_topics = engine.get_lda_topics(n_words=N_WORDS)
    print_topics(lda_topics, model_name='LDA (Gensim)')

    print("    Computing LDA coherence score...")
    lda_coherence = compute_coherence(engine.lda_model, tokenized_docs, engine.dictionary)
    print_coherence(lda_coherence, model_name='LDA')

    # ------------------------------------------------------------------
    # [5] NMF Model (Bonus comparison)
    # ------------------------------------------------------------------
    print(f"[5] Training NMF model (Bonus — LDA vs NMF comparison)...")
    engine.build_nmf(raw_texts)
    nmf_topics = engine.get_nmf_topics(n_words=N_WORDS)
    print_topics(nmf_topics, model_name='NMF (scikit-learn)')

    # ------------------------------------------------------------------
    # [6] Document-Topic assignment
    # ------------------------------------------------------------------
    print("[6] Assigning dominant topic to each document...")
    df['dominant_topic'] = engine.get_doc_topic_distribution()
    results_path = os.path.join(base_dir, 'data', 'results.csv')
    df.to_csv(results_path, index=False)
    print(f"    Results saved to {results_path}")

    # ------------------------------------------------------------------
    # [7] Save models
    # ------------------------------------------------------------------
    print("\n[7] Saving models to models/ directory...")
    models_dir = os.path.join(base_dir, 'models')
    engine.save(models_dir)
    print(f"    Models saved to {models_dir}")

    print("\n[DONE] Pipeline complete!")

if __name__ == '__main__':
    main()
