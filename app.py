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

def calculate_enhanced_score(cv_text, jd_text, skill_config):
    """
    Enhanced scoring with phrases, weighting, and frequency
    """
    if not cv_text or not jd_text:
        return 40
    
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    cv_parts = []
    jd_parts = []
    
    # Phrases (3x weight)
    for phrase in skill_config.get("phrases", []):
        cv_count = cv_lower.count(phrase.lower())
        jd_count = jd_lower.count(phrase.lower())
        cv_parts.extend([phrase] * min(cv_count * 3, 15))
        jd_parts.extend([phrase] * min(jd_count * 3, 15))
    
    # Primary keywords (2x weight)
    for kw in skill_config.get("primary", []):
        cv_count = cv_lower.count(kw.lower())
        jd_count = jd_lower.count(kw.lower())
        cv_parts.extend([kw] * min(cv_count * 2, 10))
        jd_parts.extend([kw] * min(jd_count * 2, 10))
    
    # Secondary keywords (1x weight)
    for kw in skill_config.get("secondary", []):
        cv_count = cv_lower.count(kw.lower())
        jd_count = jd_lower.count(kw.lower())
        cv_parts.extend([kw] * min(cv_count, 5))
        jd_parts.extend([kw] * min(jd_count, 5))
    
    cv_part = " ".join(cv_parts)
    jd_part = " ".join(jd_parts)
    
    # If no keywords found, return low score
    if not cv_part or not jd_part:
        return 40
    
    try:
        # Create vectorizer and calculate similarity
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vec.fit_transform([cv_part, jd_part])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        score = round(similarity * 100, 2)
        return score
    except Exception as e:
        st.warning(f"Scoring error for skill: {e}")
        return 40

def analyze_gaps(cv_text, skill_config):
    """
    Detailed gap analysis for a skill
    """
    cv_lower = cv_text.lower()
    
    analysis = {
        "missing_phrases": [],
        "present_phrases": [],
        "missing_keywords": [],
        "underused_keywords": [],
        "good_keywords": []
    }
    
    # Check phrases
    for phrase in skill_config.get("phrases", []):
        count = cv_lower.count(phrase.lower())
        if count == 0:
            analysis["missing_phrases"].append(phrase)
        else:
            analysis["present_phrases"].append((phrase, count))
    
    # Check keywords
    all_keywords = skill_config.get("primary", []) + skill_config.get("secondary", [])
    for kw in all_keywords:
        count = cv_lower.count(kw.lower())
        if count == 0:
            analysis["missing_keywords"].append(kw)
        elif count < 3:
            analysis["underused_keywords"].append((kw, count))
        else:
            analysis["good_keywords"].append((kw, count))
    
    return analysis

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
# SKILL DEFINITIONS
# ----------------------------- 

skills_enhanced = {
    "Compliance & Risk Management": {
        "phrases": ["risk management", "compliance framework", "governance system", "risk assessment"],
        "primary": ["compliance", "risk", "governance", "ethics"],
        "secondary": ["framework", "control", "oversight", "monitoring", "sustainability"],
        "threshold": 75
    },
    "Digitalization": {
        "phrases": ["digital transformation", "automated system", "technology-driven solution", "digitalization project"],
        "primary": ["digitalization", "automation", "digital transformation"],
        "secondary": ["digital", "system", "IT", "technology", "modernize", "innovation", "tool"],
        "threshold": 65
    },
    "M&A & Due Diligence": {
        "phrases": ["due diligence", "merger and acquisition", "post-merger integration", "transaction lifecycle"],
        "primary": ["merger", "acquisition", "due diligence", "transaction"],
        "secondary": ["integration", "target", "consolidation", "deal"],
        "threshold": 60
    },
    "Global Experience": {
        "phrases": ["cross-border", "international collaboration", "global team"],
        "primary": ["global", "international", "regional"],
        "secondary": ["cross-border", "headquarters", "collaboration"],
        "threshold": 70
    },
    "Project Management": {
        "phrases": ["project management", "dynamic environment", "cross-functional", "initiative ownership"],
        "primary": ["project", "program", "initiative"],
        "secondary": ["coordination", "implementation", "ownership", "priorities", "dynamic"],
        "threshold": 70
    },
    "Training": {
        "phrases": ["training program", "knowledge exchange", "capability building", "workshop"],
        "primary": ["training", "workshop", "education", "knowledge exchange"],
        "secondary": ["learning", "development", "mentoring", "coaching"],
        "threshold": 65
    },
    "Regulatory Knowledge": {
        "phrases": ["regulatory framework", "compliance requirement", "FCPA"],
        "primary": ["regulation", "FCPA", "sanctions", "framework"],
        "secondary": ["compliance", "laws", "regulatory", "medtech"],
        "threshold": 75
    }
}

# ----------------------------- 
# STREAMLIT APP
# ----------------------------- 

st.set_page_config(page_title="TalentFit Enhanced", layout="wide")
st.title("🎯 TalentFit: Enhanced Career Fit Analyzer")
st.caption("Advanced CV analysis with phrase matching, keyword weighting, and detailed gap analysis")

# File uploader
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload")

if cv_file:
    # Read CV
    with st.spinner("Reading CV..."):
        cv_text = read_pdf(cv_file)
    
    if not cv_text:
        st.error("Could not extract text from PDF. Please check the file.")
        st.stop()
    
    # Calculate scores
    with st.spinner("Analyzing CV against job requirements..."):
        results = []
        gap_analyses = {}
        
        for skill, config in skills_enhanced.items():
            score = calculate_enhanced_score(cv_text, job_desc, config)
            results.append([skill, score, config["threshold"]])
            gap_analyses[skill] = analyze_gaps(cv_text, config)
    
    df = pd.DataFrame(results, columns=["Skill", "Match %", "Threshold"])
    overall_score = round(df["Match %"].mean(), 2)
    
    # Display overall metrics
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = len(df[df["Match %"] >= df["Threshold"]])
        st.metric("Strong Matches", f"{strong_count}/{len(df)}")
    with col3:
        avg_gap = round((df["Threshold"] - df["Match %"]).clip(lower=0).mean(), 1)
        st.metric("Avg Gap to Target", f"{avg_gap}%")
    
    # Create color column for visualization
    df["Status"] = df.apply(
        lambda x: "Strong ✓" if x["Match %"] >= x["Threshold"] else "Needs Work △", 
        axis=1
    )
    
    # Visualization
    fig = px.bar(
        df, 
        x="Skill", 
        y="Match %",
        title="Skill Match Overview (with Dynamic Thresholds)",
        range_y=[0, 100],
        color="Status",
        color_discrete_map={"Strong ✓": "#00CC66", "Needs Work △": "#FF9933"}
    )
    
    # Add threshold markers
    fig.add_scatter(
        x=df["Skill"], 
        y=df["Threshold"],
        mode='markers',
        marker=dict(color='red', size=10, symbol='line-ew-open'),
        name='Target Threshold',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strengths section
    st.subheader("✅ Why Hire Me? (Key Strengths)")
    strengths_df = df[df["Match %"] >= df["Threshold"]]
    
    if strengths_df.empty:
        st.info("Focus on improvement areas below to build strengths.")
    else:
        for _, row in strengths_df.iterrows():
            gap_over = row["Match %"] - row["Threshold"]
            st.success(f"**{row['Skill']}** → {row['Match %']}% (exceeds target by {gap_over:.1f}%)")
    
    # Improvement areas
    st.subheader("🔧 Improvement Areas - Detailed Gap Analysis")
    improvement_df = df[df["Match %"] < df["Threshold"]]
    
    if improvement_df.empty:
        st.success("🎉 Excellent! All skills meet or exceed target thresholds!")
    else:
        for _, row in improvement_df.iterrows():
            skill = row['Skill']
            score = row['Match %']
            threshold = row['Threshold']
            gap_points = threshold - score
            gap = gap_analyses[skill]
            
            with st.expander(f"**{skill}** → {score}% (need +{gap_points:.1f}% to reach {threshold}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Missing High-Value Phrases:**")
                    if gap["missing_phrases"]:
                        for phrase in gap["missing_phrases"][:5]:
                            st.write(f"- *\"{phrase}\"*")
                    else:
                        st.write("✓ Excellent phrase coverage!")
                    
                    st.write("")
                    st.write("**✅ Present Phrases:**")
                    if gap["present_phrases"]:
                        for phrase, count in gap["present_phrases"][:5]:
                            st.write(f"- \"{phrase}\": {count}x ✓")
                    else:
                        st.write("No phrases detected yet.")
                
                with col2:
                    st.write("**❌ Missing Keywords:**")
                    if gap["missing_keywords"]:
                        st.write(", ".join(gap["missing_keywords"][:8]))
                    else:
                        st.write("✓ All keywords present!")
                    
                    st.write("")
                    st.write("**⚠️ Underused Keywords:**")
                    if gap["underused_keywords"]:
                        for kw, count in gap["underused_keywords"][:6]:
                            st.write(f"- *{kw}*: only {count}x")
                    else:
                        st.write("✓ Good frequency!")
                
                # Recommendations
                st.divider()
                st.write("**💡 Recommendations:**")
                
                rec_count = 1
                if gap["missing_phrases"]:
                    st.write(f"{rec_count}. Add phrase: \"{gap['missing_phrases'][0]}\"")
                    rec_count += 1
                
                if gap["underused_keywords"]:
                    st.write(f"{rec_count}. Increase: *{gap['underused_keywords'][0][0]}* (current: {gap['underused_keywords'][0][1]}x, target: 3+)")
                    rec_count += 1
                
                st.write(f"{rec_count}. Target: +{gap_points:.1f}% improvement needed")
    
    # Download results
    st.divider()
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="cv_analysis_results.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload your CV (PDF format) to begin analysis")
    
    with st.expander("ℹ️ How does scoring work?"):
        st.write("""
        **Enhanced Scoring Algorithm:**
        
        - **Phrase Matching (3x):** "digital transformation" scores higher than "digital"
        - **Primary Keywords (2x):** Core skills get double weight
        - **Secondary Keywords (1x):** Supporting terms normal weight
        - **Frequency:** 3-5 mentions shows expertise vs. 1 mention
        - **Thresholds:** Each skill has custom target based on job importance
        """)
