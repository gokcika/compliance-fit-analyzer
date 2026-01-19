import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# ----------------------------- 
# HELPER FUNCTIONS
# ----------------------------- 

def read_pdf(file):
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def calculate_similarity_simple(cv_text, jd_text, keywords):
    """
    Simple keyword-based scoring (original approach)
    No phrase weighting, no frequency caps
    """
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    # Just collect keywords that appear (with their frequency)
    cv_keywords = []
    jd_keywords = []
    
    for keyword in keywords:
        cv_count = cv_lower.count(keyword.lower())
        jd_count = jd_lower.count(keyword.lower())
        
        # Add keyword for each occurrence (no cap)
        cv_keywords.extend([keyword] * cv_count)
        jd_keywords.extend([keyword] * jd_count)
    
    cv_part = " ".join(cv_keywords) if cv_keywords else "none"
    jd_part = " ".join(jd_keywords) if jd_keywords else "none"
    
    if not cv_keywords or not jd_keywords:
        return 40
    
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([cv_part, jd_part])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(score * 100, 2)
    except:
        return 40

# ----------------------------- 
# JOB DESCRIPTION
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
    "- You are proficient in English, enabling you to collaborate effectively with global teams.\n"
    "- You are confident in decision-making under uncertainty and thrive in dynamic environments.\n"
    "- You have a strong aptitude for new technologies, digitalization, and automation.\n"
    "- You demonstrate excellent analytical and critical thinking skills.\n"
    "- You communicate effectively and build trust across diverse teams.\n"
    "- You work independently with an entrepreneurial mindset, taking ownership of projects.\n"
    "- You are a team player with strong leadership and interpersonal skills."
)

# ----------------------------- 
# SIMPLE SKILL DEFINITIONS (flat keyword lists)
# ----------------------------- 

skills_simple = {
    "Compliance & Risk Management": [
        "compliance", "risk", "ethics", "technical compliance", "sustainability", 
        "framework", "governance", "risk management", "control", "oversight", "monitoring"
    ],
    "Digitalization": [
        "digital", "digitalization", "automation", "system", "tool", "IT", 
        "technology", "modernize", "innovation", "digitization"
    ],
    "M&A & Due Diligence": [
        "merger", "acquisition", "due diligence", "integration", "transaction",
        "M&A", "target", "deal", "consolidation"
    ],
    "Global Experience": [
        "global", "regional", "international", "cross-border", "headquarters", 
        "collaboration", "worldwide"
    ],
    "Project Management": [
        "project", "program", "coordination", "initiative", "implementation", 
        "ownership", "priorities", "dynamic environment", "dynamic"
    ],
    "Training": [
        "training", "workshop", "education", "knowledge exchange", "learning", 
        "development", "mentoring", "coaching"
    ],
    "Regulatory Knowledge": [
        "regulation", "FCPA", "sanctions", "compliance", "laws", "medtech", 
        "framework", "regulatory"
    ]
}

# ----------------------------- 
# STREAMLIT APP
# ----------------------------- 

st.set_page_config(page_title="TalentFit Simple", layout="wide")
st.title("🎯 TalentFit: Simple Keyword Analyzer")
st.caption("Keyword frequency-based scoring (no phrase weighting)")

# File uploader
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload")

if cv_file:
    # Read CV
    with st.spinner("Reading CV..."):
        cv_text = read_pdf(cv_file)
    
    if not cv_text:
        st.error("Could not extract text from PDF.")
        st.stop()
    
    # Calculate scores
    with st.spinner("Analyzing..."):
        results = []
        
        for skill, keywords in skills_simple.items():
            score = calculate_similarity_simple(cv_text, job_desc, keywords)
            results.append([skill, score])
    
    df = pd.DataFrame(results, columns=["Skill", "Match %"])
    overall_score = round(df["Match %"].mean(), 2)
    
    # Display metrics
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = len(df[df["Match %"] >= 70])
        st.metric("Strong Matches (≥70%)", f"{strong_count}/{len(df)}")
    
    # Visualization
    fig = px.bar(
        df, 
        x="Skill", 
        y="Match %",
        title="Skill Match Overview (Simple Keyword Scoring)",
        range_y=[0, 100],
        color=df["Match %"].apply(lambda x: "Strong" if x >= 70 else "Needs Work"),
        color_discrete_map={"Strong": "#00CC66", "Needs Work": "#FF9933"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strengths
    st.subheader("✅ Why Hire Me? (Key Strengths)")
    strengths = df[df["Match %"] >= 70]
    
    if strengths.empty:
        st.info("No skills above 70% threshold yet.")
    else:
        for _, row in strengths.iterrows():
            st.success(f"**{row['Skill']}** → {row['Match %']}%")
    
    # Improvement areas
    st.subheader("🔧 Improvement Areas")
    improvements = df[df["Match %"] < 70]
    
    if improvements.empty:
        st.success("All skills above 70%!")
    else:
        for _, row in improvements.iterrows():
            gap = 70 - row["Match %"]
            st.warning(f"**{row['Skill']}** → {row['Match %']}% (need +{gap:.1f}%)")
    
    # Show keyword counts
    with st.expander("📊 Keyword Analysis Details"):
        for skill, keywords in skills_simple.items():
            st.write(f"**{skill}:**")
            cv_lower = cv_text.lower()
            keyword_counts = []
            for kw in keywords:
                count = cv_lower.count(kw.lower())
                if count > 0:
                    keyword_counts.append(f"{kw} ({count}x)")
            
            if keyword_counts:
                st.write(", ".join(keyword_counts))
            else:
                st.write("No keywords found")
            st.write("")
    
    # Download
    st.divider()
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        csv,
        "cv_analysis_simple.csv",
        "text/csv"
    )

else:
    st.info("👆 Upload your CV to begin")
