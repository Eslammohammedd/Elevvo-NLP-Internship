import os
import pandas as pd

from preprocess import download_nltk_resources, preprocess_text
from train import train_and_evaluate

def main():
    print("Downloading NLTK resources...")
    download_nltk_resources()
    
    print("Loading datasets from data/ directory...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'train.csv'))
    test_df  = pd.read_csv(os.path.join(base_dir, 'data', 'test.csv'))

    print("Preprocessing training data...")
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    
    print("Preprocessing test data...")
    test_df['clean_text']  = test_df['text'].apply(preprocess_text)

    print("Starting Model Training & Evaluation...")
    lr, nb, tfidf, cv = train_and_evaluate(train_df, test_df)
    
    import joblib
    print("Saving models to models/ directory...")
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(cv, os.path.join(models_dir, 'count_vectorizer.pkl'))
    joblib.dump(lr, os.path.join(models_dir, 'logistic_regression_model.pkl'))
    joblib.dump(nb, os.path.join(models_dir, 'naive_bayes_model.pkl'))
    
    print("✅ Pipeline execution complete!")

if __name__ == "__main__":
    main()
