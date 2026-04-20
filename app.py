import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

diabetes_model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
heart_model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
ckd_model = joblib.load(os.path.join(BASE_DIR, "ckd_model.pkl"))

def risk_level(p):
    if p < 0.3:
        return "Low"
    elif p < 0.7:
        return "Medium"
    else:
        return "High"

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

bgr = st.number_input("Blood Glucose (CKD)")
bu = st.number_input("Urea")
sc = st.number_input("Creatinine")
hemo = st.number_input("Hemoglobin")

if st.button("Analyze Report"):

    d_prob = diabetes_model.predict_proba([[glucose, bmi, age]])[0][1]
    h_prob = heart_model.predict_proba([[age, sex, cp, trestbps, chol, thalch, oldpeak]])[0][1]
    k_prob = ckd_model.predict_proba([[age, trestbps, bgr, bu, sc, hemo]])[0][1]

    st.subheader("Results")

    st.write(f"🩸 Diabetes: {risk_level(d_prob)} ({d_prob:.2f})")
    st.write(f"❤️ Heart Disease: {risk_level(h_prob)} ({h_prob:.2f})")
    st.write(f"🧪 Kidney Disease: {risk_level(k_prob)} ({k_prob:.2f})")
