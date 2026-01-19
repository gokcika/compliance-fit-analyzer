import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

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

# ----------------------------- 
# BALANCED KEYWORD DEFINITIONS
# ----------------------------- 

skills_balanced = {
    "Compliance & Risk Management": {
        "core": ["compliance", "compliant", "risk management", "risk mitigation"],
        "primary": ["risk", "governance", "ethics", "ethical"],
        "secondary": ["control", "oversight", "monitoring"],
        "threshold": 75
    },
    
    "Digitalization": {
        "core": ["digitalization", "digital transformation", "digitization"],
        "primary": ["automation", "automated", "automate"],
        "secondary": ["digital", "modernize", "innovation", "digitalize"],
        "generic": ["technology", "IT", "system", "tool"],
        "threshold": 65
    },
    
    "M&A & Due Diligence": {
        "core": ["merger and acquisition", "M&A", "due diligence"],
        "ma_terms": ["merger", "mergers", "acquisition", "acquisitions"],
        "dd_equivalents": [
            "due diligence",
            "legal due diligence",
            "regulatory due diligence", 
            "compliance due diligence"
        ],
        "integration": ["post-merger integration", "post-acquisition integration", "integration"],
        "transaction": ["transaction", "deal", "amalgamation", "consolidation"],
        "context": ["target", "target entity", "target company"],
        "threshold": 60
    },
    
    "Global Experience": {
        "core": ["global team", "international collaboration"],
        "primary": ["global", "international", "worldwide"],
        "secondary": ["regional", "cross-border", "headquarters"],
        "threshold": 70
    },
    
    "Project Management": {
        "core": ["project management", "program management"],
        "pm_actions": ["manage project", "lead project", "coordinate project", "execute project"],
        "primary": ["project", "program", "initiative"],
        "verbs": ["manage", "lead", "coordinate", "execute", "deliver"],
        "context": ["ownership", "priorities", "dynamic", "implementation"],
        "threshold": 70
    },
    
    "Training": {
        "core": ["training program", "knowledge exchange", "capability building"],
        "primary": ["training", "workshop", "education"],
        "secondary": ["learning", "development", "mentoring", "coaching"],
        "threshold": 65
    },
    
    "Regulatory Knowledge": {
        "core": ["regulatory framework", "regulatory requirement", "regulatory compliance"],
        "specific": ["FCPA", "Foreign Corrupt Practices Act"],
        "primary": ["regulation", "regulations", "regulatory", "compliance requirement"],
        "secondary": ["law", "legal requirement"],
        "threshold": 75
    }
}

def calculate_balanced_score(cv_text, jd_text, skill_config, skill_name):
    """
    Balanced scoring with proper weighting hierarchy
    """
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    cv_keywords = []
    jd_keywords = []
    
    weight_map = {
        "core": 5,
        "specific": 4,
        "ma_terms": 3,
        "dd_equivalents": 4,
        "pm_actions": 4,
        "primary": 2,
        "secondary": 1,
        "generic": 0.5,
        "verbs": 1,
        "context": 1
    }
    
    for category, keywords in skill_config.items():
        if category == "threshold":
            continue
        
        if not isinstance(keywords, list):
            continue
        
        weight = weight_map.get(category, 1)
        
        if weight >= 4:
            cap = 8
        elif weight >= 2:
            cap = 10
        else:
            cap = 6
        
        for kw in keywords:
            cv_count = cv_lower.count(kw.lower())
            jd_count = jd_lower.count(kw.lower())
            
            cv_keywords.extend([kw] * min(int(cv_count * weight), cap))
            jd_keywords.extend([kw] * min(int(jd_count * weight), cap))
    
    if skill_name == "M&A & Due Diligence":
        paragraphs = cv_text.split('\n')
        bonus_count = 0
        
        for para in paragraphs:
            para_lower = para.lower()
            
            has_ma = any(term in para_lower for term in ["merger", "acquisition", "m&a"])
            has_legal_dd = "legal due diligence" in para_lower or "legal diligence" in para_lower
            has_compliance = any(term in para_lower for term in ["compliance", "regulatory"])
            
            if has_ma and has_legal_dd:
                bonus_count += 2
            
            if has_ma and has_compliance and bonus_count < 4:
                bonus_count += 1
        
        cv_keywords.extend(["ma_context_bonus"] * bonus_count)
        jd_keywords.extend(["ma_context_bonus"] * 2)
    
    if skill_name == "Project Management":
        pm_phrases = [
            "manage project", "lead project", "coordinate project",
            "manage program", "lead program", "manage initiative"
        ]
        
        phrase_count = 0
        for phrase in pm_phrases:
            phrase_count += cv_lower.count(phrase)
        
        cv_keywords.extend(["pm_action_phrase"] * min(phrase_count * 3, 9))
        jd_keywords.extend(["pm_action_phrase"] * 2)
    
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
        
        for skill, config in skills_balanced.items():
            score = calculate_balanced_score(cv_text, job_desc, config, skill)
            threshold = config.get("threshold", 70)
            results.append([skill, score, threshold])
    
    df = pd.DataFrame(results, columns=["Skill", "Match %", "Threshold"])
    overall_score = round(df["Match %"].mean(), 2)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = len(df[df["Match %"] >= df["Threshold"]])
        st.metric("Strong Matches", f"{strong_count}/{len(df)}")
    
    df["Status"] = df.apply(
        lambda x: "Strong ✓" if x["Match %"] >= x["Threshold"] else "Needs Work △",
        axis=1
    )
    
    fig = px.bar(
        df, 
        x="Skill", 
        y="Match %",
        title="Skill Match Overview",
        range_y=[0, 100],
        color="Status",
        color_discrete_map={"Strong ✓": "#00CC66", "Needs Work △": "#FF9933"}
    )
    
    fig.add_scatter(
        x=df["Skill"], 
        y=df["Threshold"],
        mode='markers',
        marker=dict(color='red', size=10, symbol='line-ew-open'),
        name='Threshold',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("✅ Why Hire Me? (Key Strengths)")
    strengths = df[df["Match %"] >= df["Threshold"]]
    
    if strengths.empty:
        st.info("Focus on improvement areas below.")
    else:
        for _, row in strengths.iterrows():
            st.success(f"**{row['Skill']}** → {row['Match %']}%")
    
    st.subheader("🔧 Improvement Areas")
    improvements = df[df["Match %"] < df["Threshold"]]
    
    if improvements.empty:
        st.success("🎉 All skills meet thresholds!")
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
