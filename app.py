from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'svm_sentiment_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
dataset_candidates = [
    os.path.join(BASE_DIR, 'IMDB Dataset.csv'),
    r'C:\Users\kalpe\Desktop\sentiment_analysis\IMDB Dataset.csv',
    r'C:\Users\kalpe\IMDB Dataset.csv'
]
dataset_path = next((path for path in dataset_candidates if os.path.exists(path)), dataset_candidates[0])

# Initialize Flask app
app = Flask(__name__, template_folder='.')

def train_vectorizer(max_features=20000):
    print(f"Training and saving TF-IDF Vectorizer with {max_features} features...")

    text_data = []
    chunk_size = 10000
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        text_data.extend(chunk['review'].astype(str).tolist())

    trained_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    trained_vectorizer.fit(text_data)
    joblib.dump(trained_vectorizer, vectorizer_path)
    print(f"TF-IDF Vectorizer saved at: {vectorizer_path}")
    return trained_vectorizer

# Step 1: Train and Save TF-IDF Vectorizer if not exists
if not os.path.exists(vectorizer_path):
    vectorizer = train_vectorizer()
else:
    print("Loading existing TF-IDF Vectorizer...")
    vectorizer = joblib.load(vectorizer_path)

# Step 2: Train and Save SVM Model if not exists
if not os.path.exists(model_path):
    print("Training SVM model...")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Convert labels to numeric
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Transform text into TF-IDF vectors
    X = vectorizer.transform(df['review'])
    y = df['sentiment']

    # Train SVM model
    model = LinearSVC()
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"SVM Model saved at: {model_path}")
else:
    print("Loading existing SVM model...")
    model = joblib.load(model_path)

expected_features = getattr(model, 'n_features_in_', None)
actual_features = len(getattr(vectorizer, 'vocabulary_', {}))
if expected_features and actual_features != expected_features:
    print(
        f"Vectorizer/model mismatch found: vectorizer has {actual_features} "
        f"features, model expects {expected_features}."
    )
    vectorizer = train_vectorizer(max_features=expected_features)

print("Model and Vectorizer Loaded Successfully!")

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form['review'].strip()
        review_vector = vectorizer.transform([review_text])  # Convert to TF-IDF
        prediction = model.predict(review_vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        result_class = "positive" if prediction == 1 else "negative"

        confidence = None
        if hasattr(model, "decision_function"):
            score = abs(float(model.decision_function(review_vector)[0]))
            confidence = min(99, max(55, round(55 + (score * 18))))

        word_count = len(review_text.split())
        char_count = len(review_text)

        return render_template(
            'index.html',
            prediction_text=f'{sentiment} Review',
            sentiment=sentiment,
            result_class=result_class,
            confidence=confidence,
            word_count=word_count,
            char_count=char_count,
            review_text=review_text
        )
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
