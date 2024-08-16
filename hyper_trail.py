import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv\IMDB Dataset.csv')
texts = df['review']
labels = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create and fit the vectorizers
vectorizer_bow = CountVectorizer()
vectorizer_bow.fit(train_texts)

vectorizer_tfidf = TfidfVectorizer()
vectorizer_tfidf.fit(train_texts)

# Transform the training data
X_train_bow = vectorizer_bow.transform(train_texts)
X_train_tfidf = vectorizer_tfidf.transform(train_texts)

# Fit models
logistic_regression_bow = LogisticRegression()
logistic_regression_bow.fit(X_train_bow, train_labels)

logistic_regression_tfidf = LogisticRegression()
logistic_regression_tfidf.fit(X_train_tfidf, train_labels)

svm_bow = SVC()
svm_bow.fit(X_train_bow, train_labels)

svm_tfidf = SVC()
svm_tfidf.fit(X_train_tfidf, train_labels)

mnb_bow = MultinomialNB()
mnb_bow.fit(X_train_bow, train_labels)

mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_train_tfidf, train_labels)

# Save vectorizers
joblib.dump(vectorizer_bow, 'trained_models1/count_vectorizer.pkl')
joblib.dump(vectorizer_tfidf, 'trained_models1/tfidf_vectorizer.pkl')

# Save models
joblib.dump(logistic_regression_bow, 'trained_models1/logistic_regression_bow_model.pkl')
joblib.dump(logistic_regression_tfidf, 'trained_models1/logistic_regression_tfidf_model.pkl')
joblib.dump(svm_bow, 'trained_models1/svm_bow_model.pkl')
joblib.dump(svm_tfidf, 'trained_models1/svm_tfidf_model.pkl')
joblib.dump(mnb_bow, 'trained_models1/mnb_bow_model.pkl')
joblib.dump(mnb_tfidf, 'trained_models1/mnb_tfidf_model.pkl')
