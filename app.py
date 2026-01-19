import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2


def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text


def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)


def keyword_coverage_score(text, keywords):
    found = [k for k in keywords if k.lower() in text.lower()]
    return round((len(found) / len(keywords)) * 100, 2)


st.set_page_config(page_title="TalentFit", layout="wide")
st.title("TalentFit: Career Fit Analyzer")

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload_unique")

job_desc = (
    "Compliance digitalization, training, workshops, learning and development, "
    "and knowledge exchange are key elements of global compliance excellence."
)

skills = {
    "Training": [
        "training", "workshop", "workshops",
        "education", "learning", "development",
        "knowledge exchange", "mentoring", "coaching"
    ]
}


if cv_file:
    cv_text = read_pdf(cv_file)

    results = []

    for skill, keywords in skills.items():
        score = keyword_coverage_score(cv_text, keywords)
        results.append([skill, score])

    df = pd.DataFrame(results, columns=["Skill", "Match %"])
    st.metric("Training Score", f"{df.iloc[0]['Match %']}%")

else:
    st.info("👆 Upload your CV (PDF format) to begin")
