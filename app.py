
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import html
import unicodedata

# =========================================
# Text Normalization & Vectorizer
# =========================================
TR_STOP = {
    # kept minimal, everything else is English-only
    "ve", "ile", "bir", "için", "olan", "gibi", "de", "da", "bu", "şu", "o", "olarak"
}

def normalize_text(t: str) -> str:
    """Normalize text for stable matching: HTML unescape + Unicode normalization + strip."""
    t = html.unescape(t or "")
    t = unicodedata.normalize("NFKC", t)
    return t.strip()

# Configure a single vectorizer instance for the app
vectorizer = TfidfVectorizer(
    stop_words=list(ENGLISH_STOP_WORDS | TR_STOP),
    lowercase=True,
    ngram_range=(1, 2),
)

# =========================================
# PDF Reader
# =========================================
def read_pdf(file) -> str:
    """Extract text from a PDF safely and normalize it."""
    try:
        reader = PyPDF2.PdfReader(file)
    except Exception:
        return ""
    texts = []
    for page in getattr(reader, "pages", []):
        try:
            p = page.extract_text() or ""
        except Exception:
            p = ""
        if p:
            texts.append(p)
    return normalize_text(" ".join(texts))

# =========================================
# Similarity & Hybrid Skill Scoring
# =========================================
def cosine_similarity_percent(text1: str, text2: str) -> float:
    """Cosine similarity in [0, 100] using TF-IDF over two normalized strings."""
    text1 = (text1 or "").strip()
    text2 = (text2 or "").strip()
    if not text1 or not text2:
        return 0.0
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def skill_score(cv_text: str, jd_text: str, keywords: list[str]) -> float:
    """
    Hybrid score:
      - A) TF-IDF cosine over keyword snippets (60%)
      - B) Coverage of JD keywords found in CV (35%)
      - C) Bonus for strong phrases (max +12)
    """
    # Extract keyword-only snippets for cosine
    def snippet(text: str) -> str:
        tokens = [k for k in keywords if k.lower() in text.lower()]
        return " ".join(tokens)

    cv_part = snippet(cv_text)
    jd_part = snippet(jd_text)

    cosine_pct = cosine_similarity_percent(cv_part, jd_part) if (cv_part and jd_part) else 0.0

    # Coverage of JD keyword set
    jd_set = {k.lower() for k in keywords if k.lower() in jd_text.lower()}
    cv_set = {k.lower() for k in keywords if k.lower() in cv_text.lower()}

    coverage = 0.0
    if jd_set:
        coverage = 100.0 * len(jd_set & cv_set) / len(jd_set)

    # Phrase bonus for high-value collocations
    phrases = {
        "knowledge exchange",
        "learning & development",
        "l&d",
        "capability building",
        "capacity building",
        "e-learning",
        "microlearning",
        "blended learning",
        "instructional design",
    }
    text_lower = cv_text.lower()
    phrase_hits = sum(1 for p in phrases if p in text_lower)
    bonus = min(phrase_hits * 3.0, 12.0)  # up to +12

    score = 0.6 * cosine_pct + 0.35 * coverage + bonus
    return round(min(score, 100.0), 2)

# =========================================
# App UI
# =========================================
st.set_page_config(page_title="TalentFit", layout="wide")
st.title("TalentFit: Career Fit Analyzer")
st.caption("Analyze your CV against a fixed Siemens Healthineers job description and highlight strengths & improvement areas.")

# =========================================
# Fixed Job Description (English)
# =========================================
job_desc = normalize_text(
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

# =========================================
# Expanded Skill Keywords
# =========================================
skills = {
    "Compliance & Risk Management": [
        "compliance", "risk", "ethics", "technical compliance", "sustainability",
        "framework", "governance", "controls", "policies", "monitoring"
    ],
    "Digitalization": [
        "digital", "digitalization", "automation", "system", "tool", "it",
        "technology", "modernize", "innovation", "analytics", "dashboard", "workflow"
    ],
    "M&A & Due Diligence": [
        "merger", "acquisition", "due diligence", "integration", "transaction", "post-merger", "pmi"
    ],
    "Global Experience": [
        "global", "regional", "international", "cross-border", "headquarters", "collaboration", "multi-country"
    ],
    "Project Management": [
        "project", "program", "coordination", "initiative", "implementation",
        "ownership", "priorities", "timeline", "stakeholder", "milestone"
    ],
    "Training": [
        # Core & synonyms/phrases (expanded)
        "training", "trainings", "trainer", "workshop", "workshops",
        "learning", "learning & development", "l&d", "knowledge exchange",
        "enablement", "onboarding", "induction", "curriculum", "syllabus",
        "course", "module", "microlearning", "e-learning", "blended learning",
        "capability building", "capacity building", "coaching", "mentoring",
        "facilitation", "facilitator", "academy", "playbook", "playbooks",
        "refresher", "awareness session", "instructional design",
        "assessment", "quiz", "certification", "lms", "scorm"
    ],
    "Regulatory Knowledge": [
        "regulation", "fcpa", "sanctions", "ofac", "eu sanctions", "laws", "medtech", "framework", "uk bribery act"
    ],
}

# =========================================
# File Uploader
# =========================================
cv_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"], key="cv_upload_unique")

if cv_file:
    cv_text = read_pdf(cv_file)
    job_desc_text = job_desc  # already normalized

    # Guardrail: show a small alert if the CV looks empty
    if not cv_text:
        st.warning("We could not extract text from the PDF reliably. Please ensure your PDF is text-based (not only scanned images).")

    # Compute skill scores
    results = []
    for skill, keywords in skills.items():
        score = skill_score(cv_text, job_desc_text, keywords)
        results.append([skill, score])

    df = pd.DataFrame(results, columns=["Skill", "Match %"]).sort_values("Match %", ascending=False)
    overall_score = round(df["Match %"].mean(), 2)

    # =========================================
    # KPIs
    # =========================================
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Overall Match", f"{overall_score}%")

    with col2:
        strong_count = int((df["Match %"] >= 70).sum())
        st.metric("Strong Matches (≥70%)", f"{strong_count}/{len(df)}")

    # =========================================
    # Bar Chart
    # =========================================
    color_series = df["Match %"].apply(lambda x: "Strong" if x >= 70 else "Needs Work")
    fig = px.bar(
        df,
        x="Skill",
        y="Match %",
        title="Skill Match Overview",
        range_y=[0, 100],
        color=color_series,
        color_discrete_map={"Strong": "#00CC66", "Needs Work": "#FF9933"},
        text="Match %"
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_title="Match %", xaxis_title="Skill", uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

    # =========================================
    # Strengths & Improvements
    # =========================================
    st.subheader("✅ Key Strengths")
    strengths = df[df["Match %"] >= 70]
    if strengths.empty:
        st.info("No skills are currently ≥70%. See Improvement Areas below to raise your scores.")
    else:
        for _, row in strengths.iterrows():
            st.success(f"**{row['Skill']}** → {row['Match %']:.2f}%")

    st.subheader("🔧 Improvement Areas")
    improvements = df[df["Match %"] < 70]
    if improvements.empty:
        st.success("🎉 All skills are ≥70%!")
    else:
        for _, row in improvements.iterrows():
            st.warning(f"**{row['Skill']}** → {row['Match %']:.2f}%")

    # =========================================
    # CSV Download
    # =========================================
    st.divider()
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Detailed Results (CSV)",
        csv,
        "cv_analysis.csv",
        "text/csv",
    )

else:
    st.info("👆 Upload your CV (PDF) to begin the analysis.")
