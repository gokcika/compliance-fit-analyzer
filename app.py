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
# SEMANTIC KEYWORD DEFINITIONS
# ----------------------------- 

skills_semantic = {
    "Compliance & Risk Management": {
        "primary": ["compliance", "risk", "governance", "ethics"],
        "secondary": ["framework", "control", "oversight", "monitoring", "risk management"],
        "threshold": 75
    },
    
    "Digitalization": {
        "primary": ["digitalization", "digital transformation", "automation"],
        "secondary": ["digital", "automated", "digitalize", "automate", "modernize", 
                     "technology", "IT", "system", "tool", "innovation"],
        "related": ["tech", "technological", "digitization"],
        "threshold": 65
    },
    
    "M&A & Due Diligence": {
        "primary": ["merger", "acquisition", "M&A", "due diligence"],
        "ma_terms": ["merger", "mergers", "acquisition", "acquisitions", "M&A", "m&a"],
        "dd_variants": [
            "due diligence",
            "legal due diligence",
            "regulatory due diligence",
            "compliance due diligence",
            "diligence assessment"
        ],
        "integration": ["integration", "post-merger", "post-acquisition", "consolidation"],
        "transaction": ["transaction", "deal", "amalgamation"],
        "compliance_context": [
            "compliance risk", "compliance framework", "compliance system",
            "legal compliance", "regulatory compliance"
        ],
        "related": ["target", "restructuring"],
        "threshold": 60
    },
    
    "Global Experience": {
        "primary": ["global", "international"],
        "secondary": ["regional", "cross-border", "worldwide", "headquarters"],
        "related": ["collaboration", "collaborate"],
        "threshold": 70
    },
    
    "Project Management": {
        "primary": ["project", "program", "manage", "lead"],
        "pm_verbs": ["manage", "lead", "coordinate", "direct", "execute", "deliver"],
        "pm_nouns": ["project", "program", "initiative", "implementation"],
        "pm_context": ["ownership", "priorities", "dynamic", "cross-functional"],
        "threshold": 70
    },
    
    "Training": {
        "primary": ["training", "workshop", "education", "knowledge exchange"],
        "secondary": ["learning", "development", "mentoring", "coaching"],
        "related": ["capability building", "teach", "educator"],
        "threshold": 65
    },
    
    "Regulatory Knowledge": {
        "primary": ["regulation", "FCPA", "sanctions", "regulatory"],
        "secondary": ["framework", "requirement", "law", "compliance"],
        "threshold": 75
    }
}

def calculate_smart_score(cv_text, jd_text, skill_config, skill_name):
    """
    Smart scoring with semantic awareness
    """
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    cv_keywords = []
    jd_keywords = []
    
    for category, keywords in skill_config.items():
        if category == "threshold":
            continue
            
        if not isinstance(keywords, list):
            continue
        
        if category in ["primary", "ma_terms", "dd_variants"]:
            weight = 3
        elif category in ["compliance_context", "pm_verbs"]:
            weight = 2
        else:
            weight = 1
        
        for kw in keywords:
            cv_count = cv_lower.count(kw.lower())
            jd_count = jd_lower.count(kw.lower())
            
            cv_keywords.extend([kw] * min(cv_count * weight, 12))
            jd_keywords.extend([kw] * min(jd_count * weight, 12))
    
    # Context co-occurrence bonus for M&A
    if skill_name == "M&A & Due Diligence":
        paragraphs = cv_text.split('\n')
        for para in paragraphs:
            para_lower = para.lower()
            
            has_ma = any(term in para_lower for term in ["merger", "acquisition", "m&a", "transaction"])
            has_compliance = any(term in para_lower for term in ["compliance", "regulatory", "legal"])
            has_dd = "diligence" in para_lower or "assessment" in para_lower
            
            if has_ma and has_compliance:
                cv_keywords.extend(["ma_compliance_context"] * 5)
                jd_keywords.extend(["ma_compliance_context"] * 3)
            
            if has_ma and has_dd:
                cv_keywords.extend(["ma_diligence_context"] * 4)
                jd_keywords.extend(["ma_diligence_context"] * 3)
    
    # PM verbs + nouns proximity bonus
    if skill_name == "Project Management":
        pm_verbs = ["manage", "lead", "coordinate", "direct", "execute"]
        pm_nouns = ["project", "program", "initiative"]
        
        for verb in pm_verbs:
            for noun in pm_nouns:
                pattern = f"{verb}.{{0,50}}{noun}|{noun}.{{0,50}}{verb}"
                matches = len(re.findall(pattern, cv_lower))
                if matches > 0:
                    cv_keywords.extend(["pm_phrase"] * matches * 3)
                    jd_keywords.extend(["pm_phrase"] * 2)
    
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

st.set_page_config(page_title="TalentFit Smart", layout="wide")
st.title("🎯 TalentFit: Smart Career Fit Analyzer")
st.caption("Context-aware CV analysis with semantic keyword matching")

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload")

if cv_file:
    with st.spinner("Reading CV..."):
        cv_text = read_pdf(cv_file)
    
    if not cv_text:
        st.error("Could not extract text from PDF.")
        st.stop()
    
    with st.spinner("Analyzing with semantic matching..."):
        results = []
        
        for skill, config in skills_semantic.items():
            score = calculate_smart_score(cv_text, job_desc, config, skill)
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
            gap_over = row["Match %"] - row["Threshold"]
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
    
    with st.expander("ℹ️ What's different in Smart Scoring?"):
        st.write("""
        **Smart Semantic Algorithm Features:**
        
        1. **Synonym Recognition:** "Legal due diligence" = "Compliance due diligence" (for regulated entities)
        2. **Context Awareness:** Bonus when M&A terms + compliance terms appear in same paragraph
        3. **Phrase Proximity:** "Manage projects" scores higher than "manage" and "projects" separately
        4. **Semantic Weighting:** Related terms weighted appropriately (legal DD = equivalent to compliance DD)
        5. **Category Boosting:** Critical phrases get 3x weight, supporting terms 1-2x
        
        This prevents penalizing legitimate experience described with different (but equivalent) terminology.
        """)
