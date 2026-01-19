import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

def read_pdf(file):
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
    Simple, reliable keyword matching
    """
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    cv_keywords = []
    jd_keywords = []
    
    for keyword in keywords:
        cv_count = cv_lower.count(keyword.lower())
        jd_count = jd_lower.count(keyword.lower())
        
        # Cap at 8 to prevent single keyword domination
        cv_keywords.extend([keyword] * min(cv_count, 8))
        jd_keywords.extend([keyword] * min(jd_count, 8))
    
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
# COMPREHENSIVE KEYWORD LISTS (expanded but simple)
# ----------------------------- 

skills_final = {
    "Compliance & Risk Management": [
        # Core compliance terms
        "compliance", "compliant", "comply",
        "risk", "risks", "risk management", "risk assessment", "risk mitigation",
        "ethics", "ethical", "business ethics",
        "governance", "corporate governance",
        # Supporting terms
        "framework", "frameworks", "compliance framework",
        "control", "controls", "oversight", "monitoring",
        "sustainability", "sustainable"
    ],
    
    "Digitalization": [
        # High-value digitalization terms
        "digitalization", "digitization", "digitalize",
        "digital transformation",
        "automation", "automated", "automate", "automating",
        # Supporting digital terms
        "digital", "digitally",
        "modernize", "modernization", "modernizing",
        "innovation", "innovative", "innovate",
        # Tech terms (but less generic than before)
        "IT system", "technology solution", "digital tool"
    ],
    
    "M&A & Due Diligence": [
        # Core M&A terms
        "merger", "mergers", "merge", "merged",
        "acquisition", "acquisitions", "acquire", "acquired",
        "M&A", "m&a", "merger and acquisition",
        # Due diligence - EXPANDED with equivalents
        "due diligence",
        "legal due diligence",
        "regulatory due diligence",
        "compliance due diligence",
        "diligence assessment",
        "diligence review",
        # Integration
        "integration", "integrate", "integrated",
        "post-merger integration", "post-merger",
        "post-acquisition integration", "post-acquisition",
        # Transaction terms
        "transaction", "transactions", "transactional",
        "deal", "deals",
        "amalgamation", "amalgamate",
        "consolidation", "consolidate",
        "restructuring", "restructure",
        # Context terms
        "target", "target company", "target entity"
    ],
    
    "Global Experience": [
        "global", "globally", "global team",
        "international", "internationally", "international environment",
        "regional", "region", "regions",
        "cross-border", "cross border",
        "worldwide", "world-wide",
        "headquarters", "HQ",
        "collaboration", "collaborate", "collaborative"
    ],
    
    "Project Management": [
        # Core PM terms
        "project", "projects",
        "program", "programs", "programme",
        "project management", "program management",
        # PM actions
        "manage", "managed", "managing", "management",
        "lead", "led", "leading", "leadership",
        "coordinate", "coordinating", "coordination", "coordinator",
        "direct", "directed", "directing",
        "execute", "executed", "executing", "execution",
        "deliver", "delivered", "delivery",
        # PM context
        "initiative", "initiatives",
        "implementation", "implement", "implementing",
        "ownership", "own", "owned",
        "priorities", "prioritize", "priority",
        "dynamic", "dynamic environment", "fast-paced",
        "cross-functional"
    ],
    
    "Training": [
        # Core training terms
        "training", "trainings", "trained", "train", "trainer",
        "training program", "training programs",
        "workshop", "workshops",
        "education", "educational", "educate", "educator",
        "knowledge exchange", "knowledge sharing", "knowledge transfer",
        # Supporting terms
        "learning", "learn", "learning and development",
        "development", "develop", "developing",
        "mentoring", "mentor", "mentored",
        "coaching", "coach", "coached",
        "capability building", "capability", "upskilling",
        "teach", "teaching"
    ],
    
    "Regulatory Knowledge": [
        # Specific regulations
        "FCPA", "Foreign Corrupt Practices Act",
        "sanctions", "sanction", "sanctioned",
        "OFAC",
        # Regulatory terms
        "regulation", "regulations", "regulatory", "regulate",
        "regulatory framework", "regulatory frameworks",
        "regulatory requirement", "regulatory requirements",
        "regulatory compliance",
        "compliance requirement", "compliance requirements",
        # Legal/law terms
        "law", "laws", "legal",
        "legal requirement", "legal requirements",
        "legal compliance"
    ]
}

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
# STREAMLIT APP
# ----------------------------- 

st.set_page_config(page_title="TalentFit", layout="wide")
st.title("🎯 TalentFit: Career Fit Analyzer")
st.caption("Analyze your CV against Siemens Healthineers job requirements")

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload")

if cv_file:
    with st.spinner("Reading CV..."):
        cv_text = read_pdf(cv_file)
    
    if not cv_text:
        st.error("Could not extract text from PDF.")
        st.stop()
    
    with st.spinner("Analyzing..."):
        results = []
        
        for skill, keywords in skills_final.items():
            score = calculate_similarity_simple(cv_text, job_desc, keywords)
            results.append([skill, score])
    
    df = pd.DataFrame(results, columns=["Skill", "Match %"])
    overall_score = round(df["Match %"].mean(), 2)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = len(df[df["Match %"] >= 70])
        st.metric("Strong Matches (≥70%)", f"{strong_count}/{len(df)}")
    
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
    
    st.subheader("✅ Why Hire Me? (Key Strengths)")
    strengths = df[df["Match %"] >= 70]
    
    if strengths.empty:
        st.info("Focus on improvement areas below.")
    else:
        for _, row in strengths.iterrows():
            st.success(f"**{row['Skill']}** → {row['Match %']}%")
    
    st.subheader("🔧 Improvement Areas")
    improvements = df[df["Match %"] < 70]
    
    if improvements.empty:
        st.success("🎉 All skills above 70%!")
    else:
        for _, row in improvements.iterrows():
            st.warning(f"**{row['Skill']}** → {row['Match %']}%")
    
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
