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

def to_binary(label):
    """Map 6-class labels to binary: 0 = Fake, 1 = Real."""
    # 0=false, 1=half-true, 2=mostly-true, 3=true, 4=barely-true, 5=pants-fire
    fake_labels = {0, 4, 5}
    real_labels = {1, 2, 3}
    return 1 if label in real_labels else 0

def preprocess_text(text):
    """Full NLP preprocessing pipeline."""
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)
