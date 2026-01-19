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
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

# ----------------------------- 
# JOB DESCRIPTION (Fixed)
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
# ENHANCED SKILL DEFINITIONS
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
        "phrases": ["due diligence", "merger and acquisition", "post-merger integration", "transaction lifecycle", "M&A"],
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
# ENHANCED SCORING FUNCTION
# ----------------------------- 

def calculate_enhanced_score(cv_text, jd_text, skill_config):
    """
    Enhanced scoring with phrases, weighting, and frequency
    """
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
    
    if not cv_part or not jd_part:
        return 40
    
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([cv_part, jd_part])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(score * 100, 2)
    except:
        return 40

# ----------------------------- 
# GAP ANALYSIS FUNCTION
# ----------------------------- 

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
# STREAMLIT UI
# ----------------------------- 

st.set_page_config(page_title="TalentFit Enhanced", layout="wide")
st.title("🎯 TalentFit: Enhanced Career Fit Analyzer")
st.caption("Advanced CV analysis with phrase matching, keyword weighting, and detailed gap analysis")

# File uploader
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload")

if cv_file:
    # Read CV
    cv_text = read_pdf(cv_file)
    
    # Calculate scores
    results = []
    gap_analyses = {}
    
    for skill, config in skills_enhanced.items():
        score = calculate_enhanced_score(cv_text, job_desc, config)
        results.append([skill, score, config["threshold"]])
        gap_analyses[skill] = analyze_gaps(cv_text, config)
    
    df = pd.DataFrame(results, columns=["Skill", "Match %", "Threshold"])
    overall_score = round(df["Match %"].mean(), 2)
    
    # Display overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = len(df[df["Match %"] >= df["Threshold"]])
        st.metric("Strong Matches", f"{strong_count}/{len(df)}")
    with col3:
        avg_gap = round((df["Threshold"] - df["Match %"]).clip(lower=0).mean(), 1)
        st.metric("Avg Gap to Target", f"{avg_gap}%")
    
    # Visualization
    fig = px.bar(
        df, 
        x="Skill", 
        y="Match %",
        title="Skill Match Overview (with Dynamic Thresholds)",
        range_y=[0, 100],
        color=df.apply(lambda x: "Strong ✓" if x["Match %"] >= x["Threshold"] else "Needs Work △", axis=1),
        color_discrete_map={"Strong ✓": "#00CC66", "Needs Work △": "#FF9933"}
    )
    
    # Add threshold line for each skill
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
    
    # Improvement areas with detailed analysis
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
            
            with st.expander(f"**{skill}** → {score}% (need +{gap_points:.1f}% to reach {threshold}%) - Click for details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Missing High-Value Phrases:**")
                    if gap["missing_phrases"]:
                        for phrase in gap["missing_phrases"][:5]:
                            st.write(f"- *\"{phrase}\"*")
                    else:
                        st.write("✓ None - excellent phrase coverage!")
                    
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
                    st.write("**⚠️ Underused Keywords (add more):**")
                    if gap["underused_keywords"]:
                        for kw, count in gap["underused_keywords"][:6]:
                            st.write(f"- *{kw}*: only {count}x mention")
                    else:
                        st.write("✓ Good keyword frequency!")
                
                # Recommendations
                st.divider()
                st.write("**💡 Quick Fix Recommendations:**")
                
                rec_count = 1
                if gap["missing_phrases"]:
                    st.write(f"{rec_count}. **Add this phrase to your current role:** \"{gap['missing_phrases'][0]}\"")
                    rec_count += 1
                
                if gap["underused_keywords"]:
                    st.write(f"{rec_count}. **Increase mentions of:** *{gap['underused_keywords'][0][0]}* (currently {gap['underused_keywords'][0][1]}x, target 3+)")
                    rec_count += 1
                
                if gap["missing_keywords"]:
                    missing_sample = gap["missing_keywords"][:3]
                    st.write(f"{rec_count}. **Consider adding these keywords:** {', '.join(missing_sample)}")
                    rec_count += 1
                
                st.write(f"{rec_count}. **Target improvement:** +{gap_points:.1f}% needed to reach {threshold}% threshold")
    
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
    
    # Show example scoring criteria
    with st.expander("ℹ️ How does the scoring work?"):
        st.write("""
        **Enhanced Scoring Algorithm:**
        
        1. **Phrase Matching (3x weight):** Multi-word phrases like "digital transformation" score higher than individual words
        2. **Primary Keywords (2x weight):** Core competency keywords get double weight
        3. **Secondary Keywords (1x weight):** Supporting terms counted normally
        4. **Frequency Matters:** Mentioning keywords 3-5 times shows genuine expertise vs. one-time mention
        5. **Dynamic Thresholds:** Each skill has its own target based on job requirements
        
        **Threshold Guide:**
        - 75%: Core competencies (must be strong)
        - 70%: Important skills (should be strong)
        - 65%: Nice-to-have skills (moderate acceptable)
        - 60%: Specialized skills (basic acceptable)
        """)
