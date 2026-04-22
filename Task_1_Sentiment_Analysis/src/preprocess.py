import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def preprocess_text(text):
    """Full NLP preprocessing pipeline."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 3. Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # 4. Tokenize
    tokens = word_tokenize(text)
    # 5. Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)
