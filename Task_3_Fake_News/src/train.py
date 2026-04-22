import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_and_evaluate(train_df, test_df):
    print("TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df['clean_statement'])
    X_test  = tfidf.transform(test_df['clean_statement'])

    y_train = train_df['binary_label']
    y_test  = test_df['binary_label']

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM':          LinearSVC(max_iter=2000, random_state=42, C=1.0),
    }

    results = {}
    predictions = {}

    print("=" * 60)
    for name, clf in classifiers.items():
        print(f'\\n=== {name} ===')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')
        results[name]     = {'accuracy': acc, 'f1': f1}
        predictions[name] = y_pred
        print(f'Accuracy : {acc:.4f} ({acc*100:.2f}%)')
        print(f'F1 Score : {f1:.4f}')
        print('\\nClassification Report:')
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    print('\\n✅ Models evaluated!')
    print("=" * 60)
    
    return classifiers, tfidf
