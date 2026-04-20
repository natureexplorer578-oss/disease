import streamlit as st
import joblib
import os
import base64

# ================= BACKGROUND IMAGE =================

def set_bg():
    with open("bg.jpg", "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* Dark overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg()

# ================= LOAD MODELS =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

diabetes_model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
heart_model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
ckd_model = joblib.load(os.path.join(BASE_DIR, "ckd_model.pkl"))

# ================= FUNCTION =================

def risk_level(p):
    if p < 0.3:
        return "Low"
    elif p < 0.7:
        return "Medium"
    else:
        return "High"

# ================= UI =================

st.title("🏥 Medical Report Analyzer")

# Inputs
glucose = st.number_input("Glucose")
bmi = st.number_input("BMI")
age = st.number_input("Age")

sex = st.number_input("Sex (0/1)")
cp = st.number_input("Chest Pain (0-3)")
trestbps = st.number_input("Blood Pressure")

chol = st.number_input("Cholesterol")
thalch = st.number_input("Heart Rate")
oldpeak = st.number_input("Oldpeak")
exang = st.number_input("Exercise Induced Angina (0/1)")

bgr = st.number_input("Blood Glucose (CKD)")
bu = st.number_input("Urea")
sc = st.number_input("Creatinine")
hemo = st.number_input("Hemoglobin")

# ================= BUTTON =================

if st.button("Analyze Report"):

    d_prob = diabetes_model.predict_proba([[glucose, bmi, age]])[0][1]

    h_prob = heart_model.predict_proba([[ 
        age, sex, cp, trestbps, chol, thalch, oldpeak, exang 
    ]])[0][1]

    k_prob = ckd_model.predict_proba([[ 
        age, trestbps, bgr, bu, sc, hemo 
    ]])[0][1]

    # ================= RESULT UI =================

    st.markdown(f"""
    <div style="
        background: rgba(0,0,0,0.75);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-top: 20px;
    ">

    <h2 style="text-align:center;">🔍 Analysis Results</h2>
    <hr>

    <h3>🩸 Diabetes</h3>
    <p><b>Risk:</b> {risk_level(d_prob)}</p>
    <p><b>Score:</b> {d_prob:.2f}</p>

    <h3>❤️ Heart Disease</h3>
    <p><b>Risk:</b> {risk_level(h_prob)}</p>
    <p><b>Score:</b> {h_prob:.2f}</p>

    <h3>🧪 Kidney Disease</h3>
    <p><b>Risk:</b> {risk_level(k_prob)}</p>
    <p><b>Score:</b> {k_prob:.2f}</p>

    </div>
    """, unsafe_allow_html=True)
