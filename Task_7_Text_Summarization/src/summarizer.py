import torch
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from preprocess import download_nltk_resources

def load_t5_model(model_name='t5-small'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_t5_summary(text, tokenizer, model):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=130,
            min_length=30,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def textrank_summary(text, num_sentences=3):
    """Simple TextRank-based extractive summarization."""
    download_nltk_resources()
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    clean = lambda s: re.sub(r'[^a-z\s]', '', s.lower())
    word_freq = defaultdict(int)
    for sent in sentences:
        for word in clean(sent).split():
            if word not in stop_words and len(word) > 2:
                word_freq[word] += 1

    max_freq = max(word_freq.values()) if word_freq else 1
    word_freq = {w: f / max_freq for w, f in word_freq.items()}

    sent_scores = {}
    for sent in sentences[:30]:
        score = sum(word_freq.get(w, 0) for w in clean(sent).split())
        sent_scores[sent] = score

    top_sents = sorted(sent_scores, key=sent_scores.get, reverse=True)[:num_sentences]
    return ' '.join(s for s in sentences if s in top_sents)
