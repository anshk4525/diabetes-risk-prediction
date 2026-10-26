import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Screener", page_icon="🩺", layout="wide")

# ---------- Load the saved model (only once, then reused) ----------
BASE = Path(__file__).parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE / "models" / "lgbm_diabetes.joblib")
    with open(BASE / "models" / "model_info.json") as f:
        info = json.load(f)
    return model, info

model, info = load_model()

# ---------- Answer choices -> the codes the model expects ----------
AGE_GROUPS = {
    "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4, "40–44": 5,
    "45–49": 6, "50–54": 7, "55–59": 8, "60–64": 9, "65–69": 10,
    "70–74": 11, "75–79": 12, "80 or older": 13,
}
GEN_HEALTH = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
EDUCATION = {
    "Never attended school or kindergarten only": 1,
    "Grades 1–8": 2,
    "Grades 9–11 (some high school)": 3,
    "High school graduate": 4,
    "Some college or technical school": 5,
    "College graduate": 6,
}
INCOME = {
    "Less than $10,000": 1, "$10,000 – $15,000": 2, "$15,000 – $20,000": 3,
    "$20,000 – $25,000": 4, "$25,000 – $35,000": 5, "$35,000 – $50,000": 6,
    "$50,000 – $75,000": 7, "$75,000 or more": 8,
}
LABELS = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check in past 5 years", "Smoker": "Smoker (100+ cigarettes ever)",
    "Stroke": "Past stroke", "HeartDiseaseorAttack": "Heart disease or attack",
    "PhysActivity": "Exercise in past 30 days", "Fruits": "Fruit daily", "Veggies": "Vegetables daily",
    "HvyAlcoholConsump": "Heavy drinking", "AnyHealthcare": "Health coverage",
    "NoDocbcCost": "Skipped doctor due to cost", "DiffWalk": "Difficulty walking",
    "MentHlth": "Poor mental health days", "PhysHlth": "Poor physical health days",
}

# ---------- Custom styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg, #0F766E 0%, #0EA5E9 100%); padding: 2rem 2.2rem;
  border-radius: 18px; color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white !important; margin: 0 0 .4rem 0; font-size: 2.2rem;}
.hero p {color: #E0F2FE; margin: 0; font-size: 1.05rem;}
.stat-row {display: flex; gap: .7rem; margin-top: 1.1rem; flex-wrap: wrap;}
.stat {background: rgba(255,255,255,.18); padding: .5rem .9rem; border-radius: 10px; font-size: .9rem;}
.stat b {font-size: 1.1rem;}
.result-card {padding: 1.1rem 1.3rem; border-radius: 14px; margin: .2rem 0 1rem 0;}
.result-high {background: #FEF2F2; border: 1px solid #FECACA;}
.result-low {background: #F0FDF4; border: 1px solid #BBF7D0;}
.result-card .title {font-size: 1.15rem; font-weight: 700; margin-bottom: .25rem;}
.factor {padding: .45rem .75rem; border-radius: 8px; margin-bottom: .35rem; font-size: .95rem;}
.up {background: #FEF2F2;}
.down {background: #F0FDF4;}
.placeholder {background: white; border: 1px dashed #CBD5E1; border-radius: 14px; padding: 2rem 1.5rem;
  text-align: center; color: #475569;}
</style>
""", unsafe_allow_html=True)

# ---------- Hero header ----------
st.markdown("""
<div class="hero">
<h1>🩺 Diabetes Risk Screener</h1>
<p>Answer a few health survey questions to see whether a model trained on 250,000+ CDC survey responses would suggest diabetes screening.</p>
<div class="stat-row">
<div class="stat"><b>0.83</b> ROC-AUC</div>
<div class="stat"><b>79%</b> of cases caught</div>
<div class="stat"><b>254K</b> survey records</div>
<div class="stat"><b>LightGBM</b> model</div>
</div>
</div>
""", unsafe_allow_html=True)

tab_check, tab_model, tab_about = st.tabs(["🩺 Risk check", "📊 How the model works", "ℹ️ About"])

# =====================================================================
# TAB 1: RISK CHECK
# =====================================================================
with tab_check:
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.form("survey"):
            st.markdown("#### 👤 About you")
            c1, c2 = st.columns(2)
            age = c1.selectbox("Age group", list(AGE_GROUPS), index=6)
            sex = c2.radio("Sex", ["Female", "Male"], horizontal=True)
            bmi = c1.number_input("BMI", min_value=12, max_value=98, value=27)
            gen_health = c2.selectbox("In general, your health is…", list(GEN_HEALTH), index=2)
            education = c1.selectbox("Education", list(EDUCATION), index=5)
            income = c2.selectbox("Household income", list(INCOME), index=6)

            st.markdown("#### 🫀 Health conditions")
            c1, c2 = st.columns(2)
            high_bp = c1.checkbox("High blood pressure")
            high_chol = c1.checkbox("High cholesterol")
            stroke = c1.checkbox("Ever had a stroke")
            heart = c2.checkbox("Heart disease or heart attack")
            diff_walk = c2.checkbox("Serious difficulty walking or climbing stairs")

            st.markdown("#### 📅 Past 30 days")
            ment_hlth = st.slider("Days your mental health was not good", 0, 30, 0)
            phys_hlth = st.slider("Days your physical health was not good", 0, 30, 0)

            st.markdown("#### 🥗 Lifestyle and healthcare")
            c1, c2 = st.columns(2)
            phys_activity = c1.checkbox("Exercised in the past 30 days (outside work)", value=True)
            fruits = c1.checkbox("Eat fruit at least once a day", value=True)
            veggies = c1.checkbox("Eat vegetables at least once a day", value=True)
            smoker = c1.checkbox("Smoked at least 100 cigarettes in your life")
            heavy_drinker = c2.checkbox("Heavy drinker (men 14+ / women 7+ drinks a week)")
            any_healthcare = c2.checkbox("Have health insurance or coverage", value=True)
            chol_check = c2.checkbox("Cholesterol checked in the past 5 years", value=True)
            no_doc_cost = c2.checkbox("Couldn't see a doctor in the past year due to cost")

            submitted = st.form_submit_button("Check my risk", type="primary", use_container_width=True)

    with right:
        if not submitted:
            st.markdown("""
<div class="placeholder">
<div style="font-size:2.5rem">📋</div>
<p><b>Fill in the questions and click "Check my risk".</b></p>
<p>Your risk score, the screening recommendation, and the factors that moved your score will appear here.</p>
</div>
""", unsafe_allow_html=True)
        else:
            row = {
                "HighBP": int(high_bp), "HighChol": int(high_chol), "CholCheck": int(chol_check),
                "BMI": int(bmi), "Smoker": int(smoker), "Stroke": int(stroke),
                "HeartDiseaseorAttack": int(heart), "PhysActivity": int(phys_activity),
                "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(heavy_drinker),
                "AnyHealthcare": int(any_healthcare), "NoDocbcCost": int(no_doc_cost),
                "GenHlth": GEN_HEALTH[gen_health], "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
                "DiffWalk": int(diff_walk), "Sex": 1 if sex == "Male" else 0,
                "Age": AGE_GROUPS[age], "Education": EDUCATION[education], "Income": INCOME[income],
            }
            X = pd.DataFrame([row])[info["features"]]   # same columns, same order as training

            score = float(model.predict_proba(X)[0, 1])
            age_code = AGE_GROUPS[age]
            cutoff = info["young_threshold"] if age_code in info["young_age_codes"] else info["threshold"]

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"valueformat": ".2f", "font": {"size": 44}},
                title={"text": "Risk score", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 1], "tickvals": [0, 0.25, 0.5, 0.75, 1]},
                    "bar": {"color": "#0F172A", "thickness": 0.25},
                    "steps": [
                        {"range": [0, cutoff], "color": "#BBF7D0"},
                        {"range": [cutoff, 1], "color": "#FECACA"},
                    ],
                    "threshold": {"line": {"color": "#DC2626", "width": 4}, "thickness": 0.85, "value": cutoff},
                },
            ))
            fig.update_layout(height=250, margin=dict(l=25, r=25, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)

            # Recommendation card
            if score >= cutoff:
                st.markdown(
                    f'<div class="result-card result-high"><div class="title">⚠️ Screening suggested</div>'
                    f'Your score ({score:.2f}) is at or above the cutoff ({cutoff:.2f}). '
                    f'Consider asking a doctor about a simple diabetes blood test, such as an A1C test.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="result-card result-low"><div class="title">✅ Lower risk based on these answers</div>'
                    f'Your score ({score:.2f}) is below the cutoff ({cutoff:.2f}).</div>',
                    unsafe_allow_html=True)

            # What moved this person's score (LightGBM's built-in per-feature contributions)
            contrib = pd.Series(model.predict(X, pred_contrib=True)[0][:-1], index=info["features"])

            def describe(f):
                v = row[f]
                if f == "BMI":
                    return f"BMI: {v}"
                if f == "GenHlth":
                    return f"General health: {gen_health}"
                if f == "Age":
                    return f"Age: {age}"
                if f == "Income":
                    return f"Household income: bracket {v} of 8"
                if f == "Education":
                    return f"Education: {education}"
                if f == "Sex":
                    return f"Sex: {sex}"
                if f in ("MentHlth", "PhysHlth"):
                    return f"{LABELS[f]}: {v}"
                return f"{LABELS[f]}: {'Yes' if v else 'No'}"

            st.markdown("##### 🔍 What moved your score")
            raising = contrib[contrib > 0].sort_values(ascending=False).head(3)
            lowering = contrib[contrib < 0].sort_values().head(3)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Raising your score")
                for f in raising.index:
                    st.markdown(f'<div class="factor up">▲ {describe(f)}</div>', unsafe_allow_html=True)
            with c2:
                st.caption("Lowering your score")
                for f in lowering.index:
                    st.markdown(f'<div class="factor down">▼ {describe(f)}</div>', unsafe_allow_html=True)

            st.caption(
                "The score is a ranking score, not an exact probability: the model weighs diabetes cases more "
                "heavily so it catches more of them. Adults under 40 use a lower cutoff (0.20) because the model "
                "tends to under-score younger people. Factors show what moved this model's score, not medical causes."
            )

# =====================================================================
# TAB 2: HOW THE MODEL WORKS
# =====================================================================
with tab_model:
    st.markdown("### How good is the model?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (test)", "0.827")
    m2.metric("Cases caught (recall)", "79%")
    m3.metric("Alarms that are real", "31%")
    m4.metric("Test respondents", "50,736")
    st.write(
        "Tested on 50,736 people the model never saw during training: it caught 5,585 of 7,069 diabetes "
        "cases. Logistic regression, random forest and LightGBM all reached ROC-AUC 0.82–0.83, which suggests "
        "the survey answers themselves, not the algorithm, limit how accurate any model can be."
    )

    st.markdown("### What drives the predictions")
    st.image(str(BASE / "images" / "feature_importance.png"))
    st.write("Self-rated general health, BMI and age matter most, followed by high blood pressure and high cholesterol.")

    st.markdown("### Fairness across age groups")
    st.image(str(BASE / "images" / "recall_by_age.png"))
    st.write(
        "With one cutoff, the model caught only 39% of diabetes cases in adults under 40. "
        "Lowering their cutoff to 0.20 raised that to 64%, at the cost of more false alarms."
    )

# =====================================================================
# TAB 3: ABOUT
# =====================================================================
with tab_about:
    st.markdown("### About this project")
    st.write(
        "Built by Ansh Kumar, CS & Mathematics student at Rutgers University, as a hands-on machine learning project. "
        "Data: CDC Diabetes Health Indicators (BRFSS 2015, UCI Machine Learning Repository), 253,680 survey respondents."
    )
    st.markdown("[View the code, notebooks and full write-up on GitHub](https://github.com/anshk4525/diabetes-risk-prediction)")
    st.warning(
        "This tool is for learning purposes only. It is not medical advice and cannot diagnose diabetes. "
        "If you are worried about your health, talk to a doctor."
    )