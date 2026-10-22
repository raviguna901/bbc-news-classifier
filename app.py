
import re
import streamlit as st
import joblib
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources if missing
for resource in [
    "corpora/stopwords",
    "corpora/wordnet",
    "taggers/averaged_perceptron_tagger_eng"
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        if "stopwords" in resource:
            nltk.download("stopwords", quiet=True)
        elif "wordnet" in resource:
            nltk.download("wordnet", quiet=True)
        else:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Load saved model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_artifacts()

# Text preprocessing
sw = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

keep_words = {
    "no", "nor", "not", "never",
    "couldn't", "didn't", "doesn't", "don't",
    "isn't", "wasn't", "weren't", "won't",
    "wouldn't", "shouldn't", "haven't", "hasn't"
}

sw -= keep_words

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    words = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text)
    words = [word for word in words if word not in sw]
    tagged_words = pos_tag(words)

    cleaned_words = [
        word if word in keep_words
        else lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_words
    ]

    return " ".join(cleaned_words)

# Streamlit interface
st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 BBC News Article Classifier")
st.write("Enter a news article to predict its category.")

article = st.text_area(
    "News article",
    height=220,
    placeholder="Paste your news article here..."
)

if st.button("Classify Article", type="primary"):
    if not article.strip():
        st.warning("Please enter a news article.")
    else:
        cleaned_article = clean_text(article)

        if not cleaned_article.strip():
            st.warning("The article contains no usable words.")
        else:
            article_tfidf = tfidf.transform([cleaned_article])
            prediction = model.predict(article_tfidf)[0]

            st.success(f"Predicted category: {prediction.title()}")

            # Display model probabilities when available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(article_tfidf)[0]
                st.subheader("Category probabilities")

                for category, probability in sorted(
                    zip(model.classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                ):
                    st.write(f"**{category.title()}** — {probability:.2%}")
                    st.progress(float(probability))
