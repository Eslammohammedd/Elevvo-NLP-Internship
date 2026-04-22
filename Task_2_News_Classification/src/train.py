import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

LABEL_MAP = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

def train_and_evaluate(train_df, test_df):
    print("TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df['clean_text'])
    X_test  = tfidf.transform(test_df['clean_text'])

    y_train = train_df['label']
    y_test  = test_df['label']

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM':          LinearSVC(max_iter=2000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results = {}
    predictions = {}

    print("=" * 60)
    for name, clf in classifiers.items():
        print(f'Training {name}...')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        predictions[name] = y_pred
        print(f'  -> Accuracy: {acc:.4f} ({acc*100:.2f}%)')

    print('\\n' + '='*60)
    best_model = max(results, key=results.get)
    print(f'Best model: {best_model} (Accuracy: {results[best_model]:.4f})')
    print('Classification Report - ' + best_model)
    print(classification_report(y_test, predictions[best_model], target_names=list(LABEL_MAP.values())))
    print('='*60)
    
    return classifiers, tfidf
