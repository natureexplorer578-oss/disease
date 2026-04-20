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
 # Diabetes
    d_input = [[data["glucose"], data["bmi"], data["age"]]]
    d_prob = diabetes_model.predict_proba(d_input)[0][1]

    # Heart (✅ FIX 2: added exang)
    h_input = [[
        data["age"], data["sex"], data["cp"],
        data["trestbps"], data["chol"],
        data["thalch"], data["oldpeak"],
        data["exang"]
    ]]
    h_prob = heart_model.predict_proba(h_input)[0][1]

    # Kidney
    k_input = [[
        data["age"], data["bp"], data["bgr"],
        data["bu"], data["sc"], data["hemo"]
    ]]
    k_prob = ckd_model.predict_proba(k_input)[0][1]

   
if st.button("Analyze Report"):

    d_prob = diabetes_model.predict_proba([[glucose, bmi, age]])[0][1]
    h_prob = heart_model.predict_proba([[age, sex, cp, trestbps, chol, thalch, oldpeak]])[0][1]
    k_prob = ckd_model.predict_proba([[age, trestbps, bgr, bu, sc, hemo]])[0][1]

    st.subheader("Results")

    st.write(f"🩸 Diabetes: {risk_level(d_prob)} ({d_prob:.2f})")
    st.write(f"❤️ Heart Disease: {risk_level(h_prob)} ({h_prob:.2f})")
    st.write(f"🧪 Kidney Disease: {risk_level(k_prob)} ({k_prob:.2f})")
