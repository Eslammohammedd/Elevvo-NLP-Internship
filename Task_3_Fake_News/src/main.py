import os
import pandas as pd

from preprocess import download_nltk_resources, preprocess_text, to_binary
from train import train_and_evaluate

def main():
    print("Downloading NLTK resources...")
    download_nltk_resources()
    
    print("Loading datasets from data/ directory...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'train.csv'))
    test_df  = pd.read_csv(os.path.join(base_dir, 'data', 'test.csv'))
    
    print("Mapping labels to binary classes...")
    train_df['binary_label'] = train_df['label'].apply(to_binary)
    train_df['label_name']   = train_df['binary_label'].map({0: 'Fake', 1: 'Real'})
    
    test_df['binary_label'] = test_df['label'].apply(to_binary)
    test_df['label_name']   = test_df['binary_label'].map({0: 'Fake', 1: 'Real'})

    print("Preprocessing train set...")
    train_df['clean_statement'] = train_df['statement'].apply(preprocess_text)
    
    print("Preprocessing test set...")
    test_df['clean_statement']  = test_df['statement'].apply(preprocess_text)

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
