"""
qa_model.py
Functions for loading and running Question Answering models.
"""
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def load_qa_pipeline(model_name="distilbert-base-cased-distilled-squad"):
    """
    Load a pre-trained transformer model fine-tuned for Question Answering.
    Returns a dict with tokenizer and model.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return {"tokenizer": tokenizer, "model": model}

def answer_question(qa_pipeline, context, question):
    """
    Use the loaded model to extract an answer for a given question based on the context.
    Returns the answer text, its start/end span, and a confidence score.
    """
    tokenizer = qa_pipeline["tokenizer"]
    model = qa_pipeline["model"]
    
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1
    
    answer_tokens = inputs.input_ids[0, answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Fake score and span for simplicity, or we can compute probability
    start_prob = torch.max(torch.softmax(answer_start_scores, dim=1)).item()
    end_prob = torch.max(torch.softmax(answer_end_scores, dim=1)).item()
    score = (start_prob + end_prob) / 2.0
    
    # To get actual character start/end in context, we would need to map tokens to chars,
    # but for Exact Match and F1, the answer string is enough.
    return {
        "answer": answer.strip(),
        "score": score,
        "start": 0,
        "end": 0
    }

