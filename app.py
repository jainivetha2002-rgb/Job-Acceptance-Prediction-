# -------------------------------------------
# Job Acceptance Mini Project
# KPI Dashboard + Prediction (Streamlit)
# -------------------------------------------

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Paths (Dynamic & Safe)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==============================
# Load Data for KPIs
# ==============================
df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_job_data.csv"))

# ==============================
# KPI Calculations
# ==============================
total_candidates = len(df)
placed_count = (df["status"] == "placed").sum()
not_placed_count = (df["status"] == "not placed").sum()

placement_rate = round((placed_count / total_candidates) * 100, 2)
not_placed_rate = round((not_placed_count / total_candidates) * 100, 2)

avg_interview_score = round(
    (
        df["technical_score"]
        + df["aptitude_score"]
        + df["communication_score"]
    ).mean() / 3,
    2
)

avg_skills_match = round(df["skills_match_percentage"].mean(), 2)

high_risk_candidates = (
    (df["skills_match_percentage"] < 50)
    & (df["communication_score"] < 50)
).sum()

high_risk_percentage = round(
    (high_risk_candidates / total_candidates) * 100, 2
)

# ==============================
# Load ML Artifacts
# ==============================
model = joblib.load(os.path.join(MODEL_DIR, "job_acceptance_model.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
feature_encoders = joblib.load(os.path.join(MODEL_DIR, "feature_encoders.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(
    page_title="Job Acceptance KPI Dashboard",
    layout="wide"
)

st.title("📊 Job Acceptance KPI Dashboard")
st.write("Business Analytics + ML Prediction System")

st.divider()

# ==============================
# KPI METRICS
# ==============================
st.subheader("📌 Key Recruitment KPIs")

col1, col2, col3 = st.columns(3)
col1.metric("Total Candidates", total_candidates)
col2.metric("Placement Rate (%)", placement_rate)
col3.metric("Not Placed Rate (%)", not_placed_rate)

col4, col5, col6 = st.columns(3)
col4.metric("Avg Interview Score", avg_interview_score)
col5.metric("Avg Skills Match (%)", avg_skills_match)
col6.metric("High-Risk Candidates (%)", high_risk_percentage)

st.divider()

# ==============================
# KPI VISUALIZATIONS
# ==============================
st.subheader("📈 KPI Visual Insights")

colA, colB = st.columns(2)

with colA:
    st.write("### Placement Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="status", data=df, ax=ax1)
    st.pyplot(fig1)

with colB:
    st.write("### Company Tier vs Placement")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="company_tier", hue="status", data=df, ax=ax2)
    st.pyplot(fig2)

st.divider()

# ==============================
# PREDICTION SECTION
# ==============================
st.subheader("🔍 Predict Job Acceptance")

age = st.number_input("Age", 18, 60, 23)
gender = st.selectbox("Gender", ["male", "female"])

ssc = st.slider("SSC Percentage", 40.0, 100.0, 75.0)
hsc = st.slider("HSC Percentage", 40.0, 100.0, 78.0)
degree_pct = st.slider("Degree Percentage", 40.0, 100.0, 72.0)

degree_spec = st.selectbox(
    "Degree Specialization",
    ["computer science", "science", "commerce", "arts", "other"]
)

technical = st.slider("Technical Score", 0, 100, 70)
aptitude = st.slider("Aptitude Score", 0, 100, 65)
communication = st.slider("Communication Score", 0, 100, 68)

skills_match = st.slider("Skills Match (%)", 0, 100, 70)
certifications = st.number_input("Certifications Count", 0, 10, 1)

internship = st.selectbox("Internship Experience", ["yes", "no"])
experience_years = st.number_input("Years of Experience", 0, 20, 1)

career_switch = st.selectbox(
    "Career Switch Willingness", ["not willing", "willing"]
)
relevant_exp = st.selectbox(
    "Relevant Experience", ["relevant", "not relevant"]
)

prev_ctc = st.number_input("Previous CTC (LPA)", 0.0, 50.0, 3.0)
exp_ctc = st.number_input("Expected CTC (LPA)", 0.0, 50.0, 6.0)

company_tier = st.selectbox("Company Tier", ["tier 1", "tier 2", "tier 3"])
job_match = st.selectbox("Job Role Match", ["matched", "not matched"])

competition = st.selectbox("Competition Level", ["low", "medium", "high"])
bond = st.selectbox("Bond Requirement", ["required", "not required"])

notice = st.number_input("Notice Period (days)", 0, 180, 30)
layoff = st.selectbox("Layoff History", ["yes", "no"])

gap = st.number_input("Employment Gap (months)", 0, 60, 0)
relocation = st.selectbox("Relocation Willingness", ["willing", "not willing"])

if st.button("🚀 Predict"):
    input_data = {
        "age_years": age,
        "gender": gender,
        "ssc_percentage": ssc,
        "hsc_percentage": hsc,
        "degree_percentage": degree_pct,
        "degree_specialization": degree_spec,
        "technical_score": technical,
        "aptitude_score": aptitude,
        "communication_score": communication,
        "skills_match_percentage": skills_match,
        "certifications_count": certifications,
        "internship_experience": internship,
        "years_of_experience": experience_years,
        "career_switch_willingness": career_switch,
        "relevant_experience": relevant_exp,
        "previous_ctc_lpa": prev_ctc,
        "expected_ctc_lpa": exp_ctc,
        "company_tier": company_tier,
        "job_role_match": job_match,
        "competition_level": competition,
        "bond_requirement": bond,
        "notice_period_days": notice,
        "layoff_history": layoff,
        "employment_gap_months": gap,
        "relocation_willingness": relocation,
        "experience_category": "junior" if experience_years <= 3 else "senior",
        "academic_avg": (ssc + hsc + degree_pct) / 3,
        "academic_band": "high" if degree_pct >= 75 else "medium",
        "interview_avg": (technical + aptitude + communication) / 3,
        "interview_level": "strong" if technical >= 75 else "average",
    }

    df_input = pd.DataFrame([input_data])

    for col, encoder in feature_encoders.items():
        val = df_input[col].iloc[0]
        if val in encoder.classes_:
            df_input[col] = encoder.transform(df_input[col])
        else:
            df_input[col] = encoder.transform([encoder.classes_[0]])

    numeric_cols = df_input.select_dtypes(
        include=["int64", "float64"]
    ).columns
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    pred = model.predict(df_input)
    prob = model.predict_proba(df_input)

    label = target_encoder.inverse_transform(pred)[0]
    confidence = round(max(prob[0]) * 100, 2)

    st.success(f"Prediction: **{label.upper()}**")
    st.info(f"Confidence: **{confidence}%**")
