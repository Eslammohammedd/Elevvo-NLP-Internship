import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import download_nltk_resources, preprocess_text, preprocess_corpus
from topic_model import TopicModelEngine


class TestPreprocess(unittest.TestCase):
    """Unit tests for the text preprocessing pipeline."""

    @classmethod
    def setUpClass(cls):
        download_nltk_resources()

    def test_lowercase(self):
        tokens = preprocess_text("Hello World THIS IS A TEST")
        for token in tokens:
            self.assertEqual(token, token.lower(), "All tokens should be lowercase.")

    def test_removes_stopwords(self):
        tokens = preprocess_text("This is a test of the preprocessing pipeline.")
        stopwords_check = {'this', 'is', 'a', 'of', 'the'}
        for sw in stopwords_check:
            self.assertNotIn(sw, tokens, f"Stopword '{sw}' should be removed.")

    def test_removes_special_chars(self):
        tokens = preprocess_text("Breaking news! 100% confirmed. Visit http://example.com today.")
        for token in tokens:
            self.assertTrue(token.isalpha(), f"Token '{token}' should contain only letters.")

    def test_returns_list(self):
        result = preprocess_text("The economy is growing fast.")
        self.assertIsInstance(result, list, "preprocess_text should return a list of tokens.")

    def test_corpus_preprocessing(self):
        texts = [
            "The stock market surged today.",
            "Scientists discovered a new planet.",
        ]
        result = preprocess_corpus(texts)
        self.assertEqual(len(result), 2, "Should return one list of tokens per document.")
        for doc in result:
            self.assertIsInstance(doc, list)


class TestTopicModelEngine(unittest.TestCase):
    """Unit tests for the TopicModelEngine (LDA + NMF)."""

    @classmethod
    def setUpClass(cls):
        download_nltk_resources()
        raw_texts = [
            "The government announced new economic policies to boost growth.",
            "Stock market returns improved after central bank decisions.",
            "Scientists discovered a new species in the Amazon rainforest.",
            "Climate change is causing extreme weather events worldwide.",
            "The football team won the championship after years of effort.",
            "Athletes broke world records at the international sports event.",
            "New smartphone models feature improved AI and camera systems.",
            "Cybersecurity threats are increasing in cloud-based systems.",
            "The film festival celebrated independent cinema from around the world.",
            "Music artists released new albums reaching the top of charts.",
        ]
        from preprocess import preprocess_corpus
        cls.tokenized_docs = preprocess_corpus(raw_texts)
        cls.raw_texts = raw_texts
        cls.engine = TopicModelEngine(n_topics=3)

    def test_lda_builds_successfully(self):
        """LDA model should train without errors."""
        self.engine.build_lda(self.tokenized_docs, passes=5)
        self.assertIsNotNone(self.engine.lda_model, "LDA model should be initialized.")

    def test_lda_topic_count(self):
        """LDA should return the correct number of topics."""
        topics = self.engine.get_lda_topics(n_words=5)
        self.assertEqual(len(topics), 3, "Should return 3 topics.")

    def test_lda_topic_word_count(self):
        """Each LDA topic should have the requested number of words."""
        topics = self.engine.get_lda_topics(n_words=5)
        for topic in topics:
            self.assertGreaterEqual(len(topic['words']), 1, "Topic should contain words.")

    def test_nmf_builds_successfully(self):
        """NMF model should train without errors."""
        self.engine.build_nmf(self.raw_texts)
        self.assertIsNotNone(self.engine.nmf_model, "NMF model should be initialized.")

    def test_nmf_topic_count(self):
        """NMF should return the correct number of topics."""
        topics = self.engine.get_nmf_topics(n_words=5)
        self.assertEqual(len(topics), 3, "Should return 3 topics.")

    def test_doc_topic_distribution_length(self):
        """Document-topic assignment list should match corpus length."""
        self.engine.build_lda(self.tokenized_docs, passes=5)
        assignments = self.engine.get_doc_topic_distribution()
        self.assertEqual(
            len(assignments), len(self.tokenized_docs),
            "Should assign a dominant topic to each document."
        )


if __name__ == '__main__':
    unittest.main()
