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
        text += page.extract_text()
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

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])

# -----------------------------
# Fixed Job Description (hidden, no input required)
# -----------------------------
job_desc = """
Do you want to help create the future of healthcare? Our name, Siemens Healthineers, was selected to honor our people who dedicate their energy and passion to this cause. It reflects their pioneering spirit combined with our long history of engineering in the ever-evolving healthcare industry.

We offer you a flexible and dynamic environment with opportunities to go beyond your comfort zone in order to grow personally and professionally. Sounds interesting?

Then come and join our global team as Compliance & Digital Transformation Expert (f/m/d), to drive digital transformation in compliance and help shape the future of risk management.

Choose the best place for your work – Within the scope of this position, it is possible, in consultation with your manager, to work mobile (within Germany) up to an average volume of 60% of the respective working hours.

Even more flexibility? Mobile working from abroad is possible for up to 30 days a year under certain conditions and in selected countries.

This position can be filled anywhere in the world where Siemens Healthineers is present.

Your tasks and responsibilities:

You take ownership of developing and executing the compliance department’s digitalization strategy.
You lead and support key digitization projects, ensuring successful implementation in collaboration with global stakeholders.
You identify compliance needs together with Governance Owners and Regional Compliance Officers and turn them into impactful change projects.
You assess internal risk management processes, analyze compliance trends (e.g., technical compliance, ethics, sustainability), and develop measures to minimize risk.
You contribute to M&A transactions from due diligence to integration and support continuous improvement of the Siemens Healthineers Compliance System.
You foster knowledge exchange with compliance colleagues worldwide and drive innovation in compliance training.

Your qualifications and experience:

You have a degree in Compliance, IT, Business Administration, or a related field.
You have professional experience in compliance and/or IT and/or digitalization projects.
You have experience in project management and working in international environments.
Ideally, you have a strong understanding of risk management and compliance frameworks.

Your attributes and skills:

You are proficient in English, enabling you to collaborate effectively with global teams and communicate confidently across regions and headquarters.
You are confident in decision-making under uncertainty and thrive in dynamic environments.
You have a strong aptitude for new technologies, digitalization, and automation, enabling you to lead initiatives that modernize compliance processes and systems.
You demonstrate excellent analytical and critical thinking skills.
You communicate effectively and build trust across diverse teams ensuring smooth collaboration with governance owners, regional compliance officers, and headquarters as well as IT stakeholders.
You work independently with an entrepreneurial mindset, taking ownership of projects and managing multiple priorities in a global setting.
You are a team player wi
