import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the pickled files
with open("bow_vec_twitter.pkl", "rb") as f:
    bow_vec = pickle.load(f)

with open("twitter_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("bow_vec_reddit.pkl", "rb") as f:
    bow_vec_reddit = pickle.load(f)

with open("reddit_model.pkl", "rb") as f:
    lr_model_reddit = pickle.load(f)

def predict_emotion(text, model):
    """Predict emotion for a given text."""
    if model == "Logistic Regression (Twitter)":
        Count_text = bow_vec.transform([text])
        prediction = lr_model.predict(Count_text)
        probabilities = lr_model.predict_proba(Count_text)
    elif model == "Logistic Regression (Reddit)":
        Count_text = bow_vec_reddit.transform([text])
        prediction = lr_model_reddit.predict(Count_text)
        probabilities = lr_model_reddit.predict_proba(Count_text)
    return prediction[0], probabilities[0]

# Streamlit interface
st.image("./dash_header.jpg", width=700)
st.title("Russia Ukraine War - Emotion Detection Dashboard")
st.write("This dashboard predicts the emotion behind a text based on models trained on Twitter and Reddit data.")

model_selected = st.radio("Select a model:", ("Logistic Regression (Twitter)", "Logistic Regression (Reddit)"), index=1)
user_input = st.text_area("Enter text here, minimum of 10 words:")

if st.button("Predict"):
    prediction, probs = predict_emotion(user_input, model_selected)

    classes = lr_model.classes_ if model_selected == "Logistic Regression (Twitter)" else lr_model_reddit.classes_

    # Getting sorted indices based on probabilities
    sorted_indices = np.argsort(probs)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    st.write(f"Predicted emotion: **{prediction}**")

    prob_table = {"Emotion Class": sorted_classes, "Probability": [f"{prob:.4f}" for prob in sorted_probs]}
    st.table(prob_table)

    plt.figure(figsize=(10,5))
    plt.bar(sorted_classes, sorted_probs)
    plt.xlabel('Emotion Class')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities for each Emotion Class')
    st.pyplot(plt)



