import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# ----------------------------- 
# IMPROVED KEYWORD DEFINITIONS
# ----------------------------- 

skills_enhanced = {
    "Compliance & Risk Management": {
        "phrases": ["risk management", "compliance framework", "governance system"],
        "primary": ["compliance", "risk", "governance", "ethics"],
        "secondary": ["framework", "control", "oversight", "monitoring"],
        "threshold": 75
    },
    "Digitalization": {
        "phrases": ["digital transformation", "automated system", "technology-driven solution"],
        "primary": ["digitalization", "automation", "digital transformation"],
        "secondary": ["digital", "system", "IT", "technology", "modernize", "innovation"],
        "threshold": 65
    },
    "M&A & Due Diligence": {
        "phrases": ["due diligence", "merger and acquisition", "post-merger integration", "transaction lifecycle"],
        "primary": ["merger", "acquisition", "due diligence", "M&A"],
        "secondary": ["transaction", "integration", "target", "consolidation"],
        "threshold": 60
    },
    "Global Experience": {
        "phrases": ["cross-border", "international collaboration"],
        "primary": ["global", "international", "regional"],
        "secondary": ["cross-border", "headquarters", "collaboration"],
        "threshold": 70
    },
    "Project Management": {
        "phrases": ["project management", "dynamic environment", "cross-functional"],
        "primary": ["project", "program", "initiative"],
        "secondary": ["coordination", "implementation", "ownership", "priorities"],
        "threshold": 70
    },
    "Training": {
        "phrases": ["training program", "knowledge exchange", "workshop", "capability building"],
        "primary": ["training", "workshop", "education"],
        "secondary": ["learning", "development", "mentoring", "coaching"],
        "threshold": 65
    },
    "Regulatory Knowledge": {
        "phrases": ["regulatory framework", "compliance requirement"],
        "primary": ["regulation", "FCPA", "sanctions", "framework"],
        "secondary": ["compliance", "laws", "regulatory"],
        "threshold": 75
    }
}

# ----------------------------- 
# IMPROVED SCORING FUNCTION
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
    
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([cv_part, jd_part])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    
    return round(score * 100, 2)

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
# MAIN STREAMLIT APP
# ----------------------------- 

st.set_page_config(page_title="TalentFit Enhanced", layout="wide")
st.title("TalentFit: Enhanced Career Fit Analyzer")
st.caption("Advanced CV analysis with phrase matching, keyword weighting, and detailed gap analysis")

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_upload")

if cv_file:
    cv_text = read_pdf(cv_file)  # Your existing function
    
    results = []
    gap_analyses = {}
    
    for skill, config in skills_enhanced.items():
        score = calculate_enhanced_score(cv_text, job_desc, config)
        results.append([skill, score, config["threshold"]])
        gap_analyses[skill] = analyze_gaps(cv_text, config)
    
    df = pd.DataFrame(results, columns=["Skill", "Match %", "Threshold"])
    overall_score = round(df["Match %"].mean(), 2)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Match", f"{overall_score}%")
    with col2:
        strong_count = len(df[df["Match %"] >= df["Threshold"]])
        st.metric("Strong Matches", f"{strong_count}/{len(df)}")
    
    # Visualization
    fig = px.bar(
        df, 
        x="Skill", 
        y="Match %",
        title="Skill Match Overview (with Dynamic Thresholds)",
        range_y=[0, 100],
        color=df.apply(lambda x: "Strong" if x["Match %"] >= x["Threshold"] else "Needs Work", axis=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Strengths
    st.subheader("Why Hire Me? (Key Strengths)")
    for _, row in df[df["Match %"] >= df["Threshold"]].iterrows():
        st.success(f"✓ {row['Skill']} → {row['Match %']}% (exceeds {row['Threshold']}% threshold)")
    
    # Improvement areas with gap analysis
    st.subheader("Improvement Areas - Detailed Analysis")
    for _, row in df[df["Match %"] < df["Threshold"]].iterrows():
        skill = row['Skill']
        score = row['Match %']
        threshold = row['Threshold']
        gap = gap_analyses[skill]
        
        with st.expander(f"△ {skill} → {score}% (target: {threshold}%) - Click for details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Missing High-Value Phrases:**")
                if gap["missing_phrases"]:
                    for phrase in gap["missing_phrases"]:
                        st.write(f"- \"{phrase}\"")
                else:
                    st.write("None - good coverage!")
                
                st.write("**Present Phrases:**")
                for phrase, count in gap["present_phrases"]:
                    st.write(f"- \"{phrase}\": {count}x ✓")
            
            with col2:
                st.write("**Missing Keywords:**")
                if gap["missing_keywords"]:
                    st.write(", ".join(gap["missing_keywords"][:5]))
                else:
                    st.write("All keywords present!")
                
                st.write("**Underused Keywords (add more):**")
                for kw, count in gap["underused_keywords"][:5]:
                    st.write(f"- {kw}: only {count}x")
            
            # Recommendations
            st.write("**Quick Fix Recommendations:**")
            if gap["missing_phrases"]:
                st.write(f"1. Add phrase: \"{gap['missing_phrases'][0]}\" to your most recent role")
            if gap["underused_keywords"]:
                st.write(f"2. Increase mentions of: {gap['underused_keywords'][0][0]}")
            st.write(f"3. Target score improvement: +{threshold - score}% needed")

