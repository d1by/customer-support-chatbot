import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import nltk
from googletrans import Translator
import langid
from textblob import TextBlob
import spacy
import logging
import random

nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(filename='chatbot.log', level=logging.INFO)

data = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# print(data.head())

data = data.dropna(subset=['instruction', 'intent', 'response'])

X = data["instruction"]
y = data["intent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
with open('intent_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("\nModel and vectorizer saved successfully.")

def extract_entities(text):
    doc = nlp(text)
    return {ent.text: ent.label_ for ent in doc.ents}

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def log(user_input, response):
    logging.info(f"Query: {user_input} => Response: {response}")

def generate_response(user_input):
    input_tfidf = vectorizer.transform([user_input])
    intent = model.predict(input_tfidf)[0]
    confidence = model.predict_proba(input_tfidf).max()

    if confidence < 0.5:
        return "I didn't understand, please rephrase."
    else:
        responses = data[data['intent'] == intent]['response']
        return random.choice(responses.tolist())
    
def chatbot(user_input):
    entities = extract_entities(user_input)
    print(f"Extracted entities: {entities}")

    sentiment = analyze_sentiment(user_input)
    print(f"Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")

    response = generate_response(user_input)

    log(user_input, response)

    return response

# streamlit app
import streamlit as st

st.markdown("Hello!")

prompt = st.chat_input("Say something... ")
if prompt:
    st.write(f"You: {prompt}")
    response = chatbot(prompt)
    st.write(f"Bot: {response}")