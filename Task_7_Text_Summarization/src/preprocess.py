import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

def truncate_for_t5(text, tokenizer, max_tokens=500):
    """Prepend T5 task prefix and truncate to token limit."""
    prefix = 'summarize: '
    full_text = prefix + text
    tokens = tokenizer.encode(full_text, max_length=max_tokens + 2, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True)
