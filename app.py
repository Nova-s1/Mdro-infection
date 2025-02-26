import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("random_forest_mdro_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load feature names
with open("feature_names.pkl", "rb") as features_file:
    feature_names = pickle.load(features_file)

# Streamlit UI
st.title("MDRO Resistance Risk Prediction")
st.write("Enter patient details to predict MDRO resistance risk level.")

# Input form
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Female", "Male"])
comorbidities = st.selectbox("Comorbidities", ["None", "Diabetes", "Hypertension", "Others"])
antibiotic_exposure = st.selectbox("Antibiotic Exposure", ["No", "Yes"])

# Convert input to model format
input_data = pd.DataFrame([{   
    "Age": age,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Comorbidities_None": 1 if comorbidities == "None" else 0,
    "Antibiotic_exposure_Yes": 1 if antibiotic_exposure == "Yes" else 0,
}])

# Ensure all features exist
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Predict risk level
if st.button("Predict Risk"):
    risk_score = model.predict_proba(input_data)[:, 1][0]
    
    if risk_score < 0.3:
        risk_level = "Low Risk"
    elif risk_score < 0.7:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    st.write(f"### MDRO Resistance Risk: **{risk_level}**")
