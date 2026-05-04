from flask import Flask, render_template, request, jsonify
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'svm_sentiment_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__, template_folder='.')

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Missing model files. Run train_model.py locally before deploying.")

print("Loading TF-IDF Vectorizer...")
vectorizer = joblib.load(vectorizer_path)
print("Loading SVM model...")
model = joblib.load(model_path)
expected_features = getattr(model, 'n_features_in_', None)
actual_features = len(getattr(vectorizer, 'vocabulary_', {}))
if expected_features and actual_features != expected_features:
    raise ValueError(
        f"Vectorizer/model mismatch found: vectorizer has {actual_features} "
        f"features, model expects {expected_features}."
    )

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
