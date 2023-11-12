
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from lemmatization import lemmatize_tweet
import nltk
nltk.download('wordnet')



vectorizer = joblib.load('tfidf_vector.joblib')
model = joblib.load('multi_model.joblib')

def preprocessor(text):
    text = text.apply(lemmatize_tweet)
    tweet_transformed = vectorizer.transform(text)
    return tweet_transformed


st.title('Twitter Sentiment Analysis App')
user_input = st.text_input('Enter a text')
if user_input:
    tweet = pd.Series(user_input)
    st.write(tweet.to_list()[0])
    preprocessed_input = preprocessor(tweet)
    
    if st.button('Predict'):
        prediction = model.predict(preprocessed_input)
        if prediction == 0:
            st.write('Negative Mood')
        elif prediction == 1:
            st.write('Postive mood')
        else:
            st.write('Neutral Mood')