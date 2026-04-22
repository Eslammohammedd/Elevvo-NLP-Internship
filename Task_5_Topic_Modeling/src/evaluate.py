from gensim.models import CoherenceModel

def compute_coherence(lda_model, tokenized_docs, dictionary) -> float:
    """
    Compute U_Mass coherence score for the LDA model.
    U_Mass runs in a single thread — compatible with Python 3.13 on Windows.
    Score range: typically -14 to 0; values closer to 0 indicate better coherence.
    """
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='u_mass'
    )
    return coherence_model.get_coherence()

def print_topics(topics: list, model_name: str = 'LDA'):
    """Pretty-print topic-word distributions."""
    print('=' * 50)
    print(f'  TOP WORDS PER TOPIC — {model_name}')
    print('=' * 50)
    for t in topics:
        words_str = ' | '.join(t['words'])
        print(f"  Topic {t['topic_id'] + 1:>2}: {words_str}")
    print()

def print_coherence(score: float, model_name: str = 'LDA'):
    """Print U_Mass coherence score (range: -14 to 0; closer to 0 = better)."""
    print(f"  [{model_name}] Coherence Score (U_Mass): {score:.4f}")
    if score >= -3.0:
        label = "[GOOD]  High-quality, well-separated topics"
    elif score >= -7.0:
        label = "[FAIR]  Moderately coherent topics"
    else:
        label = "[LOW]   Topics may need more data or tuning"
    print(f"  Quality Assessment: {label}\n")
