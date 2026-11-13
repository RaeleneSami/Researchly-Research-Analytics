"""Question 2.3: Streamlit web interface for Researchly."""
from pathlib import Path
import re
import joblib
import streamlit as st
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title='Researchly Research Abstract Classifier', page_icon='📚', layout='centered')
st.title('Researchly Research Abstract Classifier')
st.write('This app classifies research abstracts as either AI-related or classical research.')

@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name('researchly_best_model.joblib')
    return joblib.load(model_path)

stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', ' ', str(text).lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(stemmer.stem(word) for word in text.split()
                    if word not in stop_words and len(word) > 2)

abstract = st.text_area('Enter research abstract:', height=220,
    placeholder='Paste the research abstract here...')
if st.button('Classify Abstract', type='primary'):
    cleaned = clean_text(abstract)
    if not cleaned:
        st.warning('Please enter a valid research abstract before classifying.')
    else:
        try:
            prediction = load_model().predict([cleaned])[0]
            display = 'artificial intelligence' if prediction == 'ai' else 'classical research'
            st.subheader('Prediction')
            st.success(f'The abstract is classified as: {display}')
        except FileNotFoundError:
            st.error('Model file missing. Run researchly_question2.py first to train and save it.')
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')
