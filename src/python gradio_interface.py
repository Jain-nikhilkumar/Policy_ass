import gradio as gr
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from transformers import BartTokenizer, BartForSequenceClassification
import joblib

# Load BART model and tokenizer
tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large')
model_bart = BartForSequenceClassification.from_pretrained('facebook/bart-large')

# Load other models and vectorizers
logistic_regression_bow = joblib.load('trained_models/logistic_regression_bow_model.pkl')
logistic_regression_tfidf = joblib.load('trained_models/logistic_regression_tfidf_model.pkl')
svm_bow = joblib.load('trained_models/svm_bow_model.pkl')
svm_tfidf = joblib.load('trained_models/svm_tfidf_model.pkl')
mnb_bow = joblib.load('trained_models/mnb_bow_model.pkl')
mnb_tfidf = joblib.load('trained_models/mnb_tfidf_model.pkl')

vectorizer_bow = joblib.load('trained_models/count_vectorizer1.pkl')
vectorizer_tfidf = joblib.load('trained_models/tfidf_vectorizer1.pkl')

def preprocess_for_others(text, vectorizer):
    return vectorizer.transform([text])

def predict_bart(text):
    inputs = tokenizer_bart(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model_bart(**inputs)
    logits = outputs.logits
    prediction = np.argmax(logits.detach().numpy(), axis=1)
    return "Positive" if prediction[0] == 1 else "Negative"

def predict_other_sentiment(text, model, vectorizer):
    try:
        preprocessed_text = vectorizer.transform([text])
        prediction = model.predict(preprocessed_text)
        return "Positive" if prediction[0] == 1 else "Negative"
    except ValueError as e:
        print(f"Error: {e}")
        # print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary)}")
        print(f"Preprocessed text shape: {preprocessed_text.shape}")
        return "Prediction error: Feature mismatch"

def gradio_interface(review, model_choice):
    if model_choice == 'BART':
        return predict_bart(review)
    elif model_choice == 'Logistic Regression (BoW)':
        return predict_other_sentiment(review, logistic_regression_bow, vectorizer_bow)
    elif model_choice == 'Logistic Regression (TF-IDF)':
        return predict_other_sentiment(review, logistic_regression_tfidf, vectorizer_tfidf)
    elif model_choice == 'SVM (BoW)':
        return predict_other_sentiment(review, svm_bow, vectorizer_bow)
    elif model_choice == 'SVM (TF-IDF)':
        return predict_other_sentiment(review, svm_tfidf, vectorizer_tfidf)
    elif model_choice == 'Naive Bayes (BoW)':
        return predict_other_sentiment(review, mnb_bow, vectorizer_bow)
    elif model_choice == 'Naive Bayes (TF-IDF)':
        # Ensure vectorizer_tfidf is loaded correctly
        if vectorizer_tfidf is not None:
            return predict_other_sentiment(review, mnb_tfidf, vectorizer_tfidf)
        else:
            return "MNB (TF-IDF) model not loaded or vectorizer missing"
    else:
        return "Invalid model choice"

# Gradio UI
def create_ui():
    gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Review", placeholder="Enter the movie review here..."),
            gr.Dropdown(
                choices=[
                    'BART',
                    'Logistic Regression (BoW)',
                    'Logistic Regression (TF-IDF)',
                    'SVM (BoW)',
                    'SVM (TF-IDF)',
                    'Naive Bayes (BoW)',
                    'Naive Bayes (TF-IDF)'
                ],
                label="Model Choice"
            )
        ],
        outputs="text",
        title="Sentiment Analysis",
        description="Select a model and input a review to get the sentiment prediction (Positive/Negative)."
    ).launch()

if __name__ == "__main__":
    create_ui()