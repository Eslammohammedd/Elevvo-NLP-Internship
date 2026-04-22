import os
import pickle
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

class TopicModelEngine:
    """
    Topic Modeling engine supporting both LDA (via Gensim) and NMF (via scikit-learn).
    Follows the same modular pattern as other pipeline tasks.
    """

    def __init__(self, n_topics: int = 5, random_state: int = 42):
        self.n_topics = n_topics
        self.random_state = random_state
        # Gensim LDA components
        self.dictionary = None
        self.lda_model  = None
        self.corpus_bow = None
        # Sklearn NMF components
        self.tfidf_vectorizer = None
        self.nmf_model        = None
        self.tfidf_matrix     = None

    # -------------------------------------------------------------------------
    # LDA (Gensim)
    # -------------------------------------------------------------------------
    def build_lda(self, tokenized_docs: list, passes: int = 15) -> models.LdaModel:
        """Train an LDA model using Gensim."""
        self.dictionary = corpora.Dictionary(tokenized_docs)
        # Filter extremes: ignore tokens that appear in <2 or >90% of docs
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        self.corpus_bow = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        self.lda_model = models.LdaModel(
            corpus=self.corpus_bow,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=self.random_state,
            passes=passes,
            alpha='auto',
            eta='auto',
        )
        return self.lda_model

    def get_lda_topics(self, n_words: int = 10) -> list:
        """Return top words for each LDA topic."""
        topics = []
        for idx, topic in self.lda_model.print_topics(num_words=n_words):
            words = [w.split('"')[1] for w in topic.split('+')]
            topics.append({'topic_id': idx, 'words': [w.strip() for w in words]})
        return topics

    def get_doc_topic_distribution(self) -> list:
        """Return dominant topic index for each document."""
        distributions = []
        for bow in self.corpus_bow:
            topic_dist = self.lda_model.get_document_topics(bow)
            dominant = max(topic_dist, key=lambda x: x[1]) if topic_dist else (0, 0.0)
            distributions.append(dominant[0])
        return distributions

    # -------------------------------------------------------------------------
    # NMF (scikit-learn) — Bonus comparison
    # -------------------------------------------------------------------------
    def build_nmf(self, raw_texts: list) -> NMF:
        """Train an NMF model using scikit-learn TF-IDF features."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            max_df=0.9,
            min_df=2,
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(raw_texts)
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=400
        )
        self.nmf_model.fit(self.tfidf_matrix)
        return self.nmf_model

    def get_nmf_topics(self, n_words: int = 10) -> list:
        """Return top words for each NMF topic."""
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = []
        for idx, component in enumerate(self.nmf_model.components_):
            top_indices = component.argsort()[-n_words:][::-1]
            words = [feature_names[i] for i in top_indices]
            topics.append({'topic_id': idx, 'words': words})
        return topics

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, models_dir: str):
        """Save trained models to disk."""
        os.makedirs(models_dir, exist_ok=True)
        if self.lda_model:
            self.lda_model.save(os.path.join(models_dir, 'lda_model.gensim'))
            self.dictionary.save(os.path.join(models_dir, 'dictionary.gensim'))
        if self.nmf_model:
            with open(os.path.join(models_dir, 'nmf_model.pkl'), 'wb') as f:
                pickle.dump(self.nmf_model, f)
            with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)

    def load_lda(self, models_dir: str):
        """Load a saved LDA model from disk."""
        self.lda_model  = models.LdaModel.load(os.path.join(models_dir, 'lda_model.gensim'))
        self.dictionary = corpora.Dictionary.load(os.path.join(models_dir, 'dictionary.gensim'))
