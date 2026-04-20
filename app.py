import streamlit as st
import joblib
import os

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

diabetes_model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
heart_model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
ckd_model = joblib.load(os.path.join(BASE_DIR, "ckd_model.pkl"))

# Risk level function
def risk_level(p):
    if p < 0.3:
        return "Low"
    elif p < 0.7:
        return "Medium"
    else:
        return "High"

# UI Title
st.title("🏥 Medical Report Analyzer")

# ================= INPUTS =================

# Diabetes inputs
glucose = st.number_input("Glucose")
bmi = st.number_input("BMI")
age = st.number_input("Age")

# Heart inputs
sex = st.number_input("Sex (0/1)")
cp = st.number_input("Chest Pain (0-3)")
trestbps = st.number_input("Blood Pressure")
chol = st.number_input("Cholesterol")
thalch = st.number_input("Heart Rate")
oldpeak = st.number_input("Oldpeak")
exang = st.number_input("Exercise Induced Angina (0/1)")

# Kidney inputs
bgr = st.number_input("Blood Glucose (CKD)")
bu = st.number_input("Urea")
sc = st.number_input("Creatinine")
hemo = st.number_input("Hemoglobin")

# ================= BUTTON =================

if st.button("Analyze Report"):

    # Diabetes (same as ipynb)
    d_input = [[glucose, bmi, age]]
    d_prob = diabetes_model.predict_proba(d_input)[0][1]

    # Heart (same order as your ipynb)
    h_input = [[
        age, sex, cp,
        trestbps, chol,
        thalch, oldpeak,
        exang
    ]]
    h_prob = heart_model.predict_proba(h_input)[0][1]

    # Kidney (same as ipynb)
    k_input = [[
        age, trestbps, bgr,
        bu, sc, hemo
    ]]
    k_prob = ckd_model.predict_proba(k_input)[0][1]

    # ================= OUTPUT =================

    st.subheader("🔍 Results")

    st.success(f"🩸 Diabetes: {risk_level(d_prob)} ({d_prob:.2f})")
    st.warning(f"❤️ Heart Disease: {risk_level(h_prob)} ({h_prob:.2f})")
    st.error(f"🧪 Kidney Disease: {risk_level(k_prob)} ({k_prob:.2f})")
