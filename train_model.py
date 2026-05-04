import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
DATASET_CANDIDATES = [
    os.path.join(BASE_DIR, "IMDB Dataset.csv"),
    r"C:\Users\kalpe\Desktop\sentiment_analysis\IMDB Dataset.csv",
    r"C:\Users\kalpe\IMDB Dataset.csv",
]


def find_dataset():
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("IMDB Dataset.csv was not found.")


def main():
    dataset_path = find_dataset()
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["review", "sentiment"])
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df = df.dropna(subset=["sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"].astype(str),
        df["sentiment"].astype(int),
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    print("Training TF-IDF vectorizer...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Linear SVM model...")
    model = LinearSVC(C=1.0, random_state=42)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=["negative", "positive"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
