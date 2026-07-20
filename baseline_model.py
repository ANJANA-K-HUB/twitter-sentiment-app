import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_tweet

def run_ml_baseline():
    df = pd.read_csv("twitter.csv")
    
    print("Cleaning text rows")
    
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    
    
    X = df['cleaned_text']
    y = df['sentiment']
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    
    print("Extracting TF-IDF text features...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # class_weight='balanced' helps counteract the heavy class imbalance you noticed!
    print("Training Logistic Regression baseline model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    
    y_pred = model.predict(X_test_tfidf)
    
    print("\n================ BASELINE EVALUATION METRICS ================")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Baseline Accuracy: {accuracy:.2%}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_ml_baseline()