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
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="TalentFit", layout="wide")
st.title("TalentFit: Career Fit Analyzer")
st.caption("Analyze your CV against a fixed Siemens Healthineers job description")

# -----------------------------
# File uploader
# -----------------------------
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload_unique")

# -----------------------------
# Fixed Job Description
# -----------------------------
job_desc = (
    "Do you want to help create the future of healthcare? Our name, Siemens Healthineers, "
    "reflects a pioneering spirit combined with a long history of engineering in healthcare.\n\n"

    "You will drive digital transformation in compliance and help shape the future of risk management.\n\n"

    "Your tasks and responsibilities:\n"
    "- Develop and execute the compliance department's digitalization strategy.\n"
    "- Lead global digitization projects and ensure successful implementation.\n"
    "- Identify compliance needs and turn them into impactful change initiatives.\n"
    "- Assess internal risk management processes and compliance trends.\n"
    "- Support M&A transactions including due diligence and integration.\n"
    "- Foster knowledge exchange with compliance colleagues worldwide.\n\n"

    "Training, workshops, learning and development, and structured knowledge exchange "
    "are essential to ensure continuous improvement of global compliance capabilities.\n\n"

    "Your qualifications:\n"
    "- Experience in compliance, IT, digitalization, and international project management.\n"
)

# -----------------------------
# Expanded Skill Keywords
# -----------------------------
skills = {
    "Compliance & Risk Management": [
        "compliance", "risk", "ethics", "framework", "governance"
    ],
    "Digitalization": [
        "digital", "digitalization", "automation", "IT", "system", "tool",
        "technology", "platform", "modernize", "innovation"
    ],
    "M&A & Due Diligence": [
        "merger", "acquisition", "M&A", "due diligence", "integration", "transaction"
    ],
    "Global Experience": [
        "global", "regional", "international", "cross-border",
        "headquarters", "collaboration"
    ],
    "Project Management": [
        "project", "program", "initiative", "implementation",
        "ownership", "priorities", "dynamic environment"
    ],
    "Training": [
        "training", "workshop", "workshops", "education",
        "learning", "development", "learning and development",
        "knowledge exchange", "capability building"
    ],
    "Regulatory Knowledge": [
        "regulation", "laws", "FCPA", 
        "medtech", "healthcare", "compliance framework"
    ]
}

# -----------------------------
# Process CV
# -----------------------------
if cv_file:
    cv_text = read_pdf(cv_file)

    results = []
    for skill, keywords in skills.items():
        cv_part = " ".join([k for k in keywords if k.lower() in cv_text.lower()])
        jd_part = " ".join([k for k in keywords if k.lower() in job_desc.lower()])

        if cv_part and jd_part:
            score = calculate_similarity(cv_part, jd_part)
        else:
            score = 40

        results.append([skill, score])

    df = pd.DataFrame(results, columns=["Skill", "Match %"])
    overall_score = round(df["Match %"].mean(), 2)

    # -----------------------------
    # Results
    # -----------------------------
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Overall Match", f"{overall_score}%")

    with col2:
        st.metric("Strong Matches (≥70%)", f"{len(df[df['Match %'] >= 70])}/{len(df)}")

    fig = px.bar(
        df,
        x="Skill",
        y="Match %",
        range_y=[0, 100],
        color=df["Match %"].apply(lambda x: "Strong" if x >= 70 else "Needs Work"),
        color_discrete_map={"Strong": "#00CC66", "Needs Work": "#FF9933"},
        title="Skill Match Overview"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload your CV (PDF format) to begin")
