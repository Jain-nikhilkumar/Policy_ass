

# 📽️ Sentiment Analysis of Movie Reviews

Welcome to the Sentiment Analysis project! This project leverages multiple machine learning models to analyze and classify movie reviews from the IMDB dataset. We have implemented and compared various models, including traditional methods and deep learning techniques.

## Task Description

- **Objective**: Develop a sentiment analysis model to classify movie reviews as positive or negative.
- **Dataset**: IMDb movie reviews.
- **Frameworks**: TensorFlow, PyTorch, scikit-learn.
  
## 🚀 Project Overview

This project includes:
- Data preprocessing
- Model training using traditional machine learning models
- Deep learning models like LSTM and BERT
- Evaluation and comparison of model performance
- Deployment using Gradio for interactive predictions

## Model Weights and Data 📂
You can access the model weights and dataset used in this project through the following Google Drive folder: Model Weights and Data[https://drive.google.com/drive/folders/1ukXTSpW3Dp2oS20pGEXms2_TESwAjd4z].
## 📋 Table of Contents

1. [Project Setup](#project-setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Training](#model-training)
   - [Traditional Models](#traditional-models)
   - [Deep Learning Models](#deep-learning-models)
4. [Model Evaluation](#model-evaluation)
5. [Gradio Interface](#gradio-interface)
6. [Usage](#usage)
7. [License](#license)

## 🛠️ Project Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

### Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

## 📊 Data Preprocessing

1. **Load Dataset:** IMDB movie reviews dataset
2. **Clean Data:**
     1.**Text Cleaning**:
   - Removing stop words.
   - Tokenization.
   - Stemming and Lemmatization.

    2. **Feature Extraction**:
   - Convert text data into numerical form using:
     - TF-IDF
     - Word2Vec
     - Embeddings
     - 
3. **Feature Extraction:**
   - Bag of Words (BoW)
   - TF-IDF
4. **Split Data:** Training and testing sets

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load IMDB dataset
data = pd.read_csv('path/to/IMDB_Dataset.csv')
reviews = data['review'].values
sentiments = np.where(data['sentiment'] == 'positive', 1, 0)

# Split data
train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(reviews, sentiments, test_size=0.25, random_state=42)
```

## 🤖 Model Training

### Traditional Models

1. **Logistic Regression**
   - **Features:** Bag of Words, TF-IDF
2. **SVM (Support Vector Machine)**
   - **Features:** Bag of Words, TF-IDF
3. **Multinomial Naive Bayes**
   - **Features:** Bag of Words, TF-IDF

### Deep Learning Models

1. **LSTM (Long Short-Term Memory)**
   - **Architecture:** LSTM with embedding layer
2. **BERT (Bidirectional Encoder Representations from Transformers)**
   - **Architecture:** Pre-trained BERT model fine-tuned on the dataset

**Sample Code for BERT Model Training:**

```python
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode data
train_encodings = tokenizer(train_reviews.tolist(), max_length=100, padding=True, truncation=True, return_tensors='tf')
test_encodings = tokenizer(test_reviews.tolist(), max_length=100, padding=True, truncation=True, return_tensors='tf')

# Build and train BERT model
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
bert_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bert_model.fit(train_encodings['input_ids'], train_sentiments, epochs=1, batch_size=16, validation_data=(test_encodings['input_ids'], test_sentiments))
```

## 📈 Model Evaluation
### Evaluation 📊

1.**Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

2.**Model Evaluation**:
  - Classification Reports
  - Confusion Matrices
**Example:**

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
predictions = model.predict(test_encodings['input_ids']).logits
predictions = tf.round(tf.nn.sigmoid(predictions))

# Evaluation
print(classification_report(test_sentiments, predictions, target_names=['Negative', 'Positive']))
print(confusion_matrix(test_sentiments, predictions))
```

## 🧩 Gradio Interface

Create a simple Gradio interface to interact with the models:

```python
import gradio as gr

def predict_sentiment(review):
    # Your model prediction code
    return prediction

iface = gr.Interface(fn=predict_sentiment, inputs="text", outputs="text")
iface.launch()
```

## 🔧 Usage

1. **Run the Gradio Interface:**

   ```bash
   python gradio_interface.py
   ```

2. **Interact with the model:** Enter a review and get the sentiment prediction.

##Challenges and Solutions 🤔
Challenge: Training deep learning models took a long time.
Solution: Used a subset of the data and reduced the number of epochs to speed up training.

Challenge: File size limits on GitHub for model weights.
Solution: Used Git Large File Storage (Git LFS) to handle and store large files.

Challenge: Issues with model GUI design and integration.
Solution: Implemented error handling and graceful degradation in the Gradio interface to manage model integration issues effectively. Ensured that the models and vectorizers are compatible and added functionality to provide meaningful feedback to users in case of errors.

##Contribution 🤝
Feel free to contribute to this project by submitting pull requests or opening issues. Your contributions can help improve the project, address any issues, and enhance its functionality.

