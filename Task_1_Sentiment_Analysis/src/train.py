import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(train_df, test_df):
    print("Vectorizing data...")
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])
    X_test_tfidf  = tfidf.transform(test_df['clean_text'])

    # CountVectorizer (for Naive Bayes)
    cv = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_cv = cv.fit_transform(train_df['clean_text'])
    X_test_cv  = cv.transform(test_df['clean_text'])

    y_train = train_df['label']
    y_test  = test_df['label']

    print("=" * 50)
    print("Training Logistic Regression (TF-IDF)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    print(f'Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)')
    print('\\nClassification Report:')
    print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

    print("=" * 50)
    print("Training Naive Bayes (CountVectorizer)...")
    nb = MultinomialNB()
    nb.fit(X_train_cv, y_train)
    y_pred_nb = nb.predict(X_test_cv)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    print(f'Accuracy: {nb_acc:.4f} ({nb_acc*100:.2f}%)')
    print('\\nClassification Report:')
    print(classification_report(y_test, y_pred_nb, target_names=['Negative', 'Positive']))
    print("=" * 50)
    
    return lr, nb, tfidf, cv
