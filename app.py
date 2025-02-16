from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Define paths
dataset_path = r'C:\Users\kalpe\Desktop\sentiment_analysis\IMDB Dataset.csv'
model_path = r'C:\Users\kalpe\Desktop\sentiment_analysis\svm_sentiment_model.pkl'
vectorizer_path = r'C:\Users\kalpe\Desktop\sentiment_analysis\tfidf_vectorizer.pkl'

# Initialize Flask app
app = Flask(__name__)

# Step 1: Train and Save TF-IDF Vectorizer if not exists
if not os.path.exists(vectorizer_path):
    print("Training and saving TF-IDF Vectorizer...")
    
    # Load dataset in chunks to handle large data
    text_data = []
    chunk_size = 10000
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
        text_data.extend(chunk['review'].astype(str).tolist())

    # Train TF-IDF Vectorizer with consistent max_features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Keep 100 features
    vectorizer.fit(text_data)
    
    # Save the vectorizer
    joblib.dump(vectorizer, vectorizer_path)
    print(f"TF-IDF Vectorizer saved at: {vectorizer_path}")
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
    model = SVC(kernel='linear')
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"SVM Model saved at: {model_path}")
else:
    print("Loading existing SVM model...")
    model = joblib.load(model_path)

print("Model and Vectorizer Loaded Successfully!")

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form['review']
        review_vector = vectorizer.transform([review_text])  # Convert to TF-IDF
        prediction = model.predict(review_vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
