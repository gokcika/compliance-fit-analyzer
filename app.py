import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# -----------------------------
# Helper functions
# -----------------------------
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="TalentFit", layout="wide")

st.title("TalentFit: Career Fit Analyzer")
st.caption("Analyze your CV against job descriptions and highlight strengths & improvement areas")

col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])

with col2:
    job_desc = st.text_area("Paste Job Description")

if cv_file and job_desc:
    cv_text = read_pdf(cv_file)

    # Skill keywords
    skills = {
        "Compliance & Risk Management": ["compliance", "risk", "ethics"],
        "Digitalization": ["digital", "automation", "system", "tool"],
        "M&A & Due Diligence": ["merger", "acquisition", "due diligence"],
        "Global Experience": ["global", "regional", "international"],
        "Project Management": ["project", "program", "coordination"],
        "Training": ["training", "workshop", "education"],
        "Regulatory Knowledge": ["regulation", "FCPA", "sanctions"]
    }

    results = []

    for skill, keywords in skills.items():
        cv_part = " ".join([k for k in keywords if k.lower() in cv_text.lower()])
        jd_part = " ".join([k for k in keywords if k.lower() in job_desc.lower()])
        score = calculate_similarity(cv_part, jd_part) if cv_part and jd_part else 40
        results.append([skill, score])

    df = pd.DataFrame(results, columns=["Skill", "Match %"])

    overall_score = round(df["Match %"].mean(), 2)

    # -----------------------------
    # Display Results
    # -----------------------------
    st.metric("Overall Match", f"{overall_score} %")

    fig = px.bar(
        df,
        x="Skill",
        y="Match %",
        title="Skill Match Overview",
        range_y=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Highlight strengths
    # -----------------------------
    st.subheader("Why Hire Me? (Key Strengths)")

    strengths = df[df["Match %"] >= 70]

    if strengths.empty:
        st.info("No particular strengths detected. Consider improving your skills.")
    else:
        for _, row in strengths.iterrows():
            st.success(f"{row['Skill']} → Strong match")

    # -----------------------------
    # Improvement Areas
    # -----------------------------
    st.subheader("Improvement Areas")

    risks = df[df["Match %"] < 70]

    if risks.empty:
        st.success("No major improvement areas detected. Strong overall fit!")
    else:
        for _, row in risks.iterrows():
            st.warning(f"{row['Skill']} → Development opportunity")
