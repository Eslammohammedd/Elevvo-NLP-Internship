"""
Generate all asset plots for Task 5 Topic Modeling.
Run: python generate_assets.py
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from preprocess import download_nltk_resources, preprocess_corpus
from topic_model import TopicModelEngine
from evaluate import compute_coherence

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']


def run():
    ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    os.makedirs(ASSETS, exist_ok=True)

    print("Downloading NLTK resources...")
    download_nltk_resources()

    print("Loading dataset...")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_news.csv')
    df = pd.read_csv(data_path)
    raw_texts = df['text'].tolist()

    print("Preprocessing...")
    tokenized_docs = preprocess_corpus(raw_texts)

    # -------------------------------------------------------------------
    # Plot 1: EDA — category distribution + text length
    # -------------------------------------------------------------------
    print("Generating eda_plots.png...")
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    cat_counts = df['category'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#F8F9FA')

    axes[0].bar(cat_counts.index, cat_counts.values,
                color=COLORS[:len(cat_counts)], edgecolor='white', linewidth=0.8)
    axes[0].set_title('Article Count by Category', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Category', labelpad=8)
    axes[0].set_ylabel('Count', labelpad=8)
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].set_facecolor('#F8F9FA')

    axes[1].hist(df['text_length'], bins=10, color='#4C72B0', edgecolor='white', rwidth=0.85)
    axes[1].set_title('Article Length Distribution (words)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Word Count', labelpad=8)
    axes[1].set_ylabel('Frequency', labelpad=8)
    axes[1].set_facecolor('#F8F9FA')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_plots.png'), bbox_inches='tight', dpi=130)
    plt.close()
    print("  -> Saved eda_plots.png")

    # -------------------------------------------------------------------
    # Plot 2: Top 20 words
    # -------------------------------------------------------------------
    print("Generating top_words.png...")
    all_tokens = [tok for doc in tokenized_docs for tok in doc]
    freq = Counter(all_tokens).most_common(20)
    words, counts = zip(*freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    bars = ax.barh(list(reversed(words)), list(reversed(counts)),
                   color='#4C72B0', edgecolor='white', linewidth=0.6)
    ax.set_title('Top 20 Most Frequent Words After Preprocessing',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Frequency', labelpad=8)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                str(int(w)), va='center', fontsize=8, color='#333')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'top_words.png'), bbox_inches='tight', dpi=130)
    plt.close()
    print("  -> Saved top_words.png")

    # -------------------------------------------------------------------
    # Train LDA + NMF
    # -------------------------------------------------------------------
    N_TOPICS = 5
    print("Training LDA model...")
    engine = TopicModelEngine(n_topics=N_TOPICS)
    engine.build_lda(tokenized_docs, passes=15)
    lda_topics = engine.get_lda_topics(n_words=10)

    print("Computing coherence (U_Mass — single-threaded)...")
    coherence = compute_coherence(engine.lda_model, tokenized_docs, engine.dictionary)
    print(f"  Coherence (U_Mass): {coherence:.4f}")

    print("Training NMF model...")
    engine.build_nmf(raw_texts)
    nmf_topics = engine.get_nmf_topics(n_words=10)

    # -------------------------------------------------------------------
    # Plot 3: LDA topics bar chart
    # -------------------------------------------------------------------
    print("Generating lda_topics.png...")
    fig, axes = plt.subplots(1, N_TOPICS, figsize=(18, 4))
    fig.patch.set_facecolor('#F8F9FA')

    for i in range(N_TOPICS):
        term_probs = engine.lda_model.show_topic(i, topn=8)
        top_ws = [w for w, _ in term_probs]
        top_ps = [p for _, p in term_probs]

        axes[i].barh(top_ws[::-1], top_ps[::-1],
                     color=COLORS[i], edgecolor='white', linewidth=0.6)
        axes[i].set_title(f'Topic {i + 1}', fontweight='bold', fontsize=11, pad=8)
        axes[i].set_xlabel('Probability', fontsize=9)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].set_facecolor('#F8F9FA')

    plt.suptitle('LDA -- Top Words per Topic', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'lda_topics.png'), bbox_inches='tight', dpi=130)
    plt.close()
    print("  -> Saved lda_topics.png")

    # -------------------------------------------------------------------
    # Plot 4: NMF topics bar chart
    # -------------------------------------------------------------------
    print("Generating nmf_topics.png...")
    feature_names = engine.tfidf_vectorizer.get_feature_names_out()

    fig, axes = plt.subplots(1, N_TOPICS, figsize=(18, 4))
    fig.patch.set_facecolor('#F8F9FA')

    for i, component in enumerate(engine.nmf_model.components_):
        top_idx = component.argsort()[-8:][::-1]
        top_ws = [feature_names[j] for j in top_idx]
        top_ps = [component[j] for j in top_idx]

        axes[i].barh(top_ws[::-1], top_ps[::-1],
                     color=COLORS[i], edgecolor='white', linewidth=0.6)
        axes[i].set_title(f'Topic {i + 1}', fontweight='bold', fontsize=11, pad=8)
        axes[i].set_xlabel('Weight', fontsize=9)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].set_facecolor('#F8F9FA')

    plt.suptitle('NMF -- Top Words per Topic', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'nmf_topics.png'), bbox_inches='tight', dpi=130)
    plt.close()
    print("  -> Saved nmf_topics.png")

    # -------------------------------------------------------------------
    # Plot 5: Document-topic distribution
    # -------------------------------------------------------------------
    print("Generating topic_distribution.png...")
    df['dominant_topic'] = engine.get_doc_topic_distribution()
    df['dominant_topic_label'] = df['dominant_topic'].apply(lambda x: f'Topic {x + 1}')
    topic_dist = df['dominant_topic_label'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    ax.bar(topic_dist.index, topic_dist.values,
           color=COLORS[:len(topic_dist)], edgecolor='white', linewidth=0.8)
    ax.set_title('Document Distribution by Dominant LDA Topic',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Topic', labelpad=8)
    ax.set_ylabel('Number of Articles', labelpad=8)
    for i, v in enumerate(topic_dist.values):
        ax.text(i, v + 0.1, str(v), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'topic_distribution.png'), bbox_inches='tight', dpi=130)
    plt.close()
    print("  -> Saved topic_distribution.png")

    # -------------------------------------------------------------------
    # Plot 6: LDA vs NMF comparison
    # -------------------------------------------------------------------
    print("Generating model_comparison.png...")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    x = np.arange(N_TOPICS)
    width = 0.35

    lda_vals = [len(t['words']) for t in lda_topics]
    nmf_vals = [len(t['words']) for t in nmf_topics]

    ax.bar(x - width / 2, lda_vals, width, label='LDA', color='#4C72B0', edgecolor='white')
    ax.bar(x + width / 2, nmf_vals, width, label='NMF', color='#DD8452', edgecolor='white')

    ax.set_title('LDA vs NMF -- Words Extracted per Topic',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Topic', labelpad=8)
    ax.set_ylabel('Top Words Count', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Topic {i + 1}' for i in range(N_TOPICS)])
    ax.legend()
    ax.set_ylim(0, max(max(lda_vals), max(nmf_vals)) + 3)

    ax.text(0.98, 0.97,
            f'LDA Coherence (U_Mass): {coherence:.4f}',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8EFF7', edgecolor='#4C72B0'))

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'model_comparison.png'), bbox_inches='tight', dpi=130)
    plt.close()
    print("  -> Saved model_comparison.png")

    print("\nAll assets generated successfully!")
    print(f"Location: {ASSETS}")


if __name__ == '__main__':
    run()
