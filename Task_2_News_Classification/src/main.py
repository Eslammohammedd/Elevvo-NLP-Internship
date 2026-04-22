import os
import pandas as pd

from preprocess import download_nltk_resources, preprocess_text
from train import train_and_evaluate, LABEL_MAP

def main():
    print("Downloading NLTK resources...")
    download_nltk_resources()
    
    print("Loading datasets from data/ directory...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'train.csv'))
    test_df  = pd.read_csv(os.path.join(base_dir, 'data', 'test.csv'))
    
    train_df['category'] = train_df['label'].map(LABEL_MAP)
    test_df['category']  = test_df['label'].map(LABEL_MAP)

    print("Preprocessing train set...")
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    
    print("Preprocessing test set...")
    test_df['clean_text']  = test_df['text'].apply(preprocess_text)

    print("Starting Model Training & Evaluation...")
    classifiers, tfidf = train_and_evaluate(train_df, test_df)
    
    import joblib
    print("Saving models to models/ directory...")
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    for name, clf in classifiers.items():
        safe_name = name.replace(' ', '_').replace('/', '_').lower()
        joblib.dump(clf, os.path.join(models_dir, f'{safe_name}_model.pkl'))
    
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    main()
