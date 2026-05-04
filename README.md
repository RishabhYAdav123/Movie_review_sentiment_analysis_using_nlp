# 🎬 Movie Review Sentiment Analysis using NLP

A machine learning-based web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** using Natural Language Processing (NLP).

🌐 **Live Demo:**
👉 https://movie-review-sentiment-analysis-using-nlp.onrender.com/

---

## 🚀 Project Overview

This project focuses on building an end-to-end NLP pipeline that processes raw text reviews and classifies sentiment using machine learning techniques. It demonstrates how real-world text data can be transformed into meaningful insights.

---

## ✨ Features

* 🔍 Text preprocessing (tokenization, stopword removal, cleaning)
* 📊 Feature extraction using TF-IDF
* 🤖 Sentiment classification using ML models
* 🌐 Web interface for real-time prediction
* 📈 Model evaluation (Accuracy, Precision, Recall, F1-score)
* ⚡ Lightweight deployment using Flask

---

## 🧠 How It Works

1. User inputs a movie review
2. Text is preprocessed (cleaned and tokenized)
3. Converted into numerical features using TF-IDF
4. Passed to trained ML model
5. Output: **Positive 😊** or **Negative 😞**

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Libraries:**

  * Scikit-learn
  * NLTK / SpaCy
  * NumPy
* **Model:** Logistic Regression / Naïve Bayes
* **Deployment:** Flask + Render

---

## 📂 Project Structure

```
├── app.py
├── model.pkl
├── vectorizer.pkl
├── templates/
├── static/
├── train.py
├── predict.py
├── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/RishabhYAdav123/Movie_review_sentiment_analysis_using_nlp.git
cd Movie_review_sentiment_analysis_using_nlp
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Application

```
python app.py
```

---

## 💻 Usage

### 👉 Web App

Open in browser:

```
http://localhost:5000
```

### 👉 CLI Prediction

```
python predict.py --review "This movie was fantastic!"
```

---

## 📊 Model Performance

* Accuracy: ~85–90% (depending on dataset)
* Uses TF-IDF + ML classifier

---

## ⚠️ Deployment Note

* Hosted on Render (free tier)
* First request may take **30–50 seconds** due to cold start

---

## 🔮 Future Improvements

* Integrate BERT / Transformers
* Add confidence score
* Improve UI/UX
* Multi-language support
* Real-time analytics dashboard

---

## 👨‍💻 Author

**Rishabh Yadav**
BTech CSE (AI & ML)

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
