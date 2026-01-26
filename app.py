
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import numpy as np

# -----------------------------
# Helper functions
# -----------------------------
def read_pdf(file):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(file)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None
        if page_text:
            text += page_text + " "
    return text

def split_sentences(text: str):
    """Split text into sentences."""
    # Normalize any stray HTML escapes (defensive)
    text = text.replace("&amp;", "&")
    # Split on punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s and s.strip()]

def extract_sentences(cv_text, keywords):
    """Extract CV sentences that include any of the keywords."""
    sentences = split_sentences(cv_text)
    relevant = []
    for s in sentences:
        s_low = s.lower()
        if any(k.lower() in s_low for k in keywords):
            relevant.append(s)
    return relevant

def calculate_similarity(text1, text2):
    """Compute similarity using TF-IDF and cosine similarity."""
    if not text1 or not text2 or not text1.strip() or not text2.strip():
        return None  # Not mentioned
    # English-only stop words (sklearn's built-in)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="TalentFit", layout="wide")
st.title("TalentFit: Career Fit Analyzer")
st.caption("Analyze your PDF CV against a fixed Siemens Healthineers job description and highlight strengths & improvement areas.")

# -----------------------------
# File uploader
# -----------------------------
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload_unique")

# -----------------------------
# Job Description (Expander)
# -----------------------------
job_desc = (
    "Do you want to help create the future of healthcare? Our name, Siemens Healthineers, "
    "was selected to honor our people who dedicate their energy and passion to this cause. "
    "It reflects their pioneering spirit combined with our long history of engineering "
    "in the ever-evolving healthcare industry.\n\n"
    "Responsibilities include driving compliance and risk governance, supporting digitalization "
    "initiatives, contributing to M&A due diligence, enabling global collaboration, leading "
    "project management activities, delivering training programs, and applying regulatory "
    "knowledge across medtech contexts."
)

with st.expander("📄 Job Description"):
    st.text(job_desc)

# -----------------------------
# Skill keywords WITH WEIGHTS
# -----------------------------
skills = {
    "Compliance & Risk Management": {
        "keywords": ["compliance", "risk", "ethics", "technical compliance", "sustainability", "framework", "governance"],
        "weight": 1.0
    },
    "Digitalization": {
        "keywords": ["digital", "digitalization", "automation", "system", "tool", "IT", "technology", "modernize", "innovation"],
        "weight": 1.0
    },
    "M&A & Due Diligence": {
        "keywords": ["merger", "acquisition", "due diligence", "integration", "transaction"],
        "weight": 1.0
    },
    "Global Experience": {
        "keywords": ["global", "regional", "international", "cross-border", "headquarters", "collaboration"],
        "weight": 1.0
    },
    "Project Management": {
        "keywords": ["project", "program", "coordination", "initiative", "implementation", "ownership", "priorities", "dynamic environment"],
        "weight": 1.0
    },
    "Training": {
        "keywords": [
            "training", "workshop", "education", "knowledge exchange", "learning", "development",
            "teach", "teaching", "instructor", "facilitation", "coaching", "mentor", "mentoring",
            "onboard", "onboarding", "curriculum", "program design", "skill development", "capacity building",
            "knowledge transfer", "knowledge sharing", "train the trainer", "upskilling"
        ],
        "weight": 1.5
    },
    "Regulatory Knowledge": {
        "keywords": ["regulation", "FCPA", "sanctions", "compliance", "laws", "medtech", "framework"],
        "weight": 1.0
    }
}

# -----------------------------
# Process CV
# -----------------------------
if cv_file:
    cv_text = read_pdf(cv_file)

    if not cv_text or not cv_text.strip():
        st.error("No text could be extracted from the PDF. Please upload a text-based (non-scanned) PDF.")
        st.stop()

    cv_sentences = split_sentences(cv_text)

    results = []
    total_weight = 0.0
    weighted_sum = 0.0
    skill_sentences = {}

    # Pre-split JD and select relevant parts by keywords to make similarity fairer
    jd_sentences = split_sentences(job_desc)

    for skill, skill_data in skills.items():
        keywords = skill_data["keywords"]
        weight = skill_data["weight"]

        # Sentences in CV that match the keywords
        relevant_sents = extract_sentences(cv_text, keywords)
        skill_sentences[skill] = relevant_sents

        # Relevant sentences from JD that match the keywords (fallback to full JD if none)
        jd_relevant = [s for s in jd_sentences if any(k.lower() in s.lower() for k in keywords)]
        jd_part = " ".join(jd_relevant) if jd_relevant else job_desc

        cv_part = " ".join(relevant_sents)

        score = calculate_similarity(cv_part, jd_part)
        if score is None:
            display_score = "Not mentioned"
            score_for_mean = 0.0
            example_sentence = "No example in CV"
        else:
            display_score = score
            score_for_mean = float(score)
            example_sentence = relevant_sents[0] if relevant_sents else "No example in CV"

        weighted_sum += score_for_mean * weight
        total_weight += weight

        results.append([skill, display_score, score_for_mean, weight, example_sentence])

    df = pd.DataFrame(results, columns=["Skill", "Match %", "Score Numeric", "Weight", "Example Sentence"])
    df_sorted = df.sort_values("Score Numeric", ascending=False).reset_index(drop=True)
    overall_score = round((weighted_sum / total_weight), 2) if total_weight > 0 else 0.0

    # -----------------------------
    # Metrics
    # -----------------------------
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = int((df["Score Numeric"] >= 70).sum())
        st.metric("Strong Matches (≥70%)", f"{strong_count}/{len(df)}")
    with col3:
        if not df_sorted.empty:
            top_skill_row = df_sorted.iloc[0]
            st.metric("Top Skill", f"{top_skill_row['Skill']}", f"{top_skill_row['Match %']}")
        else:
            st.metric("Top Skill", "—", "—")

    # -----------------------------
    # Skills Ranked Table
    # -----------------------------
    st.subheader("📊 Skills Ranked by Match")
    for idx, row in df_sorted.iterrows():
        c1, c2, c3 = st.columns([0.5, 3, 3])
        with c1:
            st.markdown(f"**#{idx+1}**")
        with c2:
            st.markdown(f"**{row['Skill']}**")
        with c3:
            val = row['Match %']
            example = row["Example Sentence"]
            if val == "Not mentioned":
                st.info(f"N/A — {example}")
            elif row['Score Numeric'] >= 70:
                st.success(f"{val}% — {example}")
            else:
                st.warning(f"{val}% — {example}")

    # -----------------------------
    # Strengths
    # -----------------------------
    st.subheader("✅ Key Strengths")
    strengths = df_sorted[df_sorted["Score Numeric"] >= 70]
    if strengths.empty:
        st.info("Focus on improvement areas below.")
    else:
        for _, row in strengths.iterrows():
            sentences = skill_sentences[row["Skill"]]
            example = sentences[0] if sentences else "No example in CV"
            st.success(f"**{row['Skill']}** → {row['Match %']}%\n> {example}")

    # -----------------------------
    # Improvement Areas
    # -----------------------------
    st.subheader("🔧 Improvement Areas")
    improvements = df_sorted[df_sorted["Score Numeric"] < 70]
    if improvements.empty:
        st.success("🎉 All skills are at or above 70%!")
    else:
        for _, row in improvements.iterrows():
            missing_keywords = [
                k for k in skills[row["Skill"]]["keywords"]
                if not any(k.lower() in s.lower() for s in cv_sentences)
            ]
            suggestion = f"Consider mentioning: {', '.join(missing_keywords)}" if missing_keywords else "Add relevant experience."
            st.warning(f"**{row['Skill']}** → {row['Match %']}%\n> {suggestion}")

    # -----------------------------
    # CSV Download
    # -----------------------------
    st.divider()
    csv = df_sorted.drop(columns=["Score Numeric"]).to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        csv,
        "cv_analysis.csv",
        "text/csv"
    )

else:
    st.info("👆 Upload your CV (PDF format) to begin.")
