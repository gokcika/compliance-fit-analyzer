import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import numpy as np
import openai  # eski SDK uyumlu

# -----------------------------
# OpenAI API Key
# -----------------------------
# Streamlit Cloud'da st.secrets içinde OPENAI_API_KEY olmalı
openai.api_key = st.secrets.get("OPENAI_API_KEY")

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

def get_embedding(text):
    """Return 1536-dim embedding using OpenAI embeddings API."""
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response['data'][0]['embedding'])

def calculate_similarity(text1, text2):
    """Sentence-level similarity using embeddings."""
    if not text1.strip() or not text2.strip():
        return None  # Not mentioned
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    score = cosine_similarity([emb1], [emb2])[0][0]
    return round(score * 100, 2)

def split_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_sentences(cv_text, keywords):
    sentences = split_sentences(cv_text)
    relevant = []
    for s in sentences:
        for k in keywords:
            if k.lower() in s.lower():
                relevant.append(s)
                break
    return relevant

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="TalentFit", layout="wide")
st.title("TalentFit: Career Fit Analyzer")
st.caption("Analyze your CV against a fixed Siemens Healthineers job description and highlight strengths & improvement areas")

# -----------------------------
# File uploader
# -----------------------------
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload_unique")

# -----------------------------
# Job Description (expander)
# -----------------------------
job_desc = (
    "Do you want to help create the future of healthcare? Our name, Siemens Healthineers, "
    "was selected to honor our people who dedicate their energy and passion to this cause. "
    "It reflects their pioneering spirit combined with our long history of engineering "
    "in the ever-evolving healthcare industry.\n\n"

    "We offer you a flexible and dynamic environment with opportunities to go beyond your comfort zone "
    "in order to grow personally and professionally. Sounds interesting?\n\n"

    "Then come and join our global team as Compliance & Digital Transformation Expert (f/m/d), "
    "to drive digital transformation in compliance and help shape the future of risk management.\n\n"

    "Choose the best place for your work – Within the scope of this position, it is possible, in consultation "
    "with your manager, to work mobile (within Germany) up to an average volume of 60% of the respective working hours.\n\n"

    "Even more flexibility? Mobile working from abroad is possible for up to 30 days a year under certain conditions "
    "and in selected countries.\n\n"

    "This position can be filled anywhere in the world where Siemens Healthineers is present.\n\n"

    "Your tasks and responsibilities:\n"
    "- You take ownership of developing and executing the compliance department's digitalization strategy.\n"
    "- You lead and support key digitization projects, ensuring successful implementation in collaboration with global stakeholders.\n"
    "- You identify compliance needs together with Governance Owners and Regional Compliance Officers and turn them into impactful change projects.\n"
    "- You assess internal risk management processes, analyze compliance trends (e.g., technical compliance, ethics, sustainability), and develop measures to minimize risk.\n"
    "- You contribute to M&A transactions from due diligence to integration and support continuous improvement of the Siemens Healthineers Compliance System.\n"
    "- You foster knowledge exchange with compliance colleagues worldwide and drive innovation in compliance training.\n\n"

    "Your qualifications and experience:\n"
    "- You have a degree in Compliance, IT, Business Administration, or a related field.\n"
    "- You have professional experience in compliance and/or IT and/or digitalization projects.\n"
    "- You have experience in project management and working in international environments.\n"
    "- Ideally, you have a strong understanding of risk management and compliance frameworks.\n\n"

    "Your attributes and skills:\n"
    "- You are proficient in English, enabling you to collaborate effectively with global teams and communicate confidently across regions and headquarters.\n"
    "- You are confident in decision-making under uncertainty and thrive in dynamic environments.\n"
    "- You have a strong aptitude for new technologies, digitalization, and automation, enabling you to lead initiatives that modernize compliance processes and systems.\n"
    "- You demonstrate excellent analytical and critical thinking skills.\n"
    "- You communicate effectively and build trust across diverse teams ensuring smooth collaboration with governance owners, regional compliance officers, and headquarters as well as IT stakeholders.\n"
    "- You work independently with an entrepreneurial mindset, taking ownership of projects and managing multiple priorities in a global setting.\n"
    "- You are a team player with strong leadership and interpersonal skills."
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
            "teach", "teaching", "instructor", "facilitation", "facilitating", "coaching", "mentor", "mentoring",
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
    cv_sentences = split_sentences(cv_text)

    results = []
    total_weight = 0
    weighted_sum = 0
    skill_sentences = {}  # Skill → list of CV sentences

    for skill, skill_data in skills.items():
        keywords = skill_data["keywords"]
        weight = skill_data["weight"]

        relevant_sents = extract_sentences(cv_text, keywords)
        skill_sentences[skill] = relevant_sents

        jd_part = " ".join([k for k in keywords if k.lower() in job_desc.lower()])
        cv_part = " ".join(relevant_sents)

        score = calculate_similarity(cv_part, jd_part)
        if score is None:
            display_score = "Not mentioned"
            score_for_mean = 0
        else:
            display_score = score
            score_for_mean = score

        weighted_sum += score_for_mean * weight
        total_weight += weight

        example_sentence = relevant_sents[0] if relevant_sents else "No example in CV"
        results.append([skill, display_score, score_for_mean, weight, example_sentence])

    df = pd.DataFrame(results, columns=["Skill", "Match %", "Score Numeric", "Weight", "Example Sentence"])
    df_sorted = df.sort_values("Score Numeric", ascending=False).reset_index(drop=True)
    overall_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0

    # -----------------------------
    # Three Column Metrics
    # -----------------------------
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = len(df[df["Score Numeric"] >= 70])
        st.metric("Strong Matches (≥70%)", f"{strong_count}/{len(df)}")
    with col3:
        top_skill_row = df_sorted.iloc[0]
        st.metric("Top Skill", f"{top_skill_row['Skill'][:20]}...", f"{top_skill_row['Match %']}%")

    # -----------------------------
    # Skills Ranked Table
    # -----------------------------
    st.subheader("📊 Skills Ranked by Match")
    for idx, row in df_sorted.iterrows():
        col1, col2, col3 = st.columns([0.5, 3, 3])
        with col1:
            st.markdown(f"**#{idx+1}**")
        with col2:
            st.markdown(f"**{row['Skill']}**")
        with col3:
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
    st.subheader("✅ Why Hire Me? (Key Strengths)")
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
        st.success("🎉 All skills above 70%!")
    else:
        for _, row in improvements.iterrows():
            missing_keywords = [k for k in skills[row["Skill"]]["keywords"] if not any(k.lower() in s.lower() for s in cv_sentences)]
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
    st.info("👆 Upload your CV (PDF format) to begin")
