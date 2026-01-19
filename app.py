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
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="TalentFit", layout="wide")
st.title("TalentFit: Career Fit Analyzer")
st.caption("Analyze your CV against a fixed Siemens Healthineers job description and highlight strengths & improvement areas")

# -----------------------------
# File uploader with unique key
# -----------------------------
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload_unique")

# -----------------------------
# Fixed Job Description (Python-friendly)
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

# -----------------------------
# Skill keywords
# -----------------------------
skills = {
    "Compliance & Risk Management": [
        "compliance", "risk", "ethics", "technical compliance", "sustainability", "framework", "governance"
    ],
    "Digitalization": [
        "digital", "digitalization", "automation", "system", "tool", "IT", "technology", "modernize", "innovation"
    ],
    "M&A & Due Diligence": [
        "merger", "acquisition", "due diligence", "integration", "transaction"
    ],
    "Global Experience": [
        "global", "regional", "international", "cross-border", "headquarters", "collaboration"
    ],
    "Project Management": [
        "project", "program", "coordination", "initiative", "implementation", "ownership", "priorities", "dynamic environment"
    ],
    "Training": [
        "training", "workshop", "education", "knowledge exchange", "learning", "development", "trainings", "professional development"
        "knowledge exchange", "knowledge sharing", "capability building"
    ],
    "Regulatory Knowledge": [
        "regulation", "FCPA", "sanctions", "compliance", "laws", "medtech", "framework"
    ]
}

# -----------------------------
# Process CV and calculate skill match
# -----------------------------
if cv_file:
    cv_text = read_pdf(cv_file)

    results = []
    for skill, keywords in skills.items():
        cv_part = " ".join([k for k in keywords if k.lower() in cv_text.lower()])
        jd_part = " ".join([k for k in keywords if k.lower() in job_desc.lower()])
        score = calculate_similarity(cv_part, jd_part) if cv_part and jd_part else 40
        results.append([skill, score])

    df = pd.DataFrame(results, columns=["Skill", "Match %"])
    overall_score = round(df["Match %"].mean(), 2)

    # -----------------------------
    # Display Results - Two Column Layout
    # -----------------------------
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    
    with col2:
        strong_count = len(df[df["Match %"] >= 70])
        st.metric("Strong Matches (≥70%)", f"{strong_count}/{len(df)}")

    # -----------------------------
    # Colored Bar Chart
    # -----------------------------
    fig = px.bar(
        df,
        x="Skill",
        y="Match %",
        title="Skill Match Overview",
        range_y=[0, 100],
        color=df["Match %"].apply(lambda x: "Strong" if x >= 70 else "Needs Work"),
        color_discrete_map={"Strong": "#00CC66", "Needs Work": "#FF9933"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Highlight strengths
    # -----------------------------
    st.subheader("✅ Why Hire Me? (Key Strengths)")
    strengths = df[df["Match %"] >= 70]

    if strengths.empty:
        st.info("Focus on improvement areas below.")
    else:
        for _, row in strengths.iterrows():
            st.success(f"**{row['Skill']}** → {row['Match %']}%")

    # -----------------------------
    # Improvement Areas
    # -----------------------------
    st.subheader("🔧 Improvement Areas")
    improvements = df[df["Match %"] < 70]

    if improvements.empty:
        st.success("🎉 All skills above 70%!")
    else:
        for _, row in improvements.iterrows():
            st.warning(f"**{row['Skill']}** → {row['Match %']}%")

    # -----------------------------
    # CSV Download
    # -----------------------------
    st.divider()
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        csv,
        "cv_analysis.csv",
        "text/csv"
    )

else:
    st.info("👆 Upload your CV (PDF format) to begin")
