import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

model = load_model()

st.title("💰 Loan Approval Prediction")
st.markdown("Predict loan approval likelihood using a tuned **XGBoost** model with **SHAP explainability**.")

st.header("Applicant Details")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income (monthly)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=10)
    loan_term = st.selectbox("Loan Term (months)", [120, 180, 240, 300, 360], index=4)
    credit_history = st.selectbox("Credit History (1 = good, 0 = bad)", [1, 0])

if st.button("Predict Loan Approval", type="primary"):
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Loan likely to be **APPROVED** (Confidence: {proba*100:.1f}%)")
    else:
        st.error(f"❌ Loan likely to be **REJECTED** (Confidence: {(1-proba)*100:.1f}%)")

    st.progress(float(proba))

    # SHAP explanation for this prediction
    st.subheader("Why this prediction? (SHAP Explanation)")
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    input_transformed = preprocessor.transform(input_df)
    feature_names = (
        ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
         'Credit_History', 'Dependents'] +
        ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    )

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(input_transformed)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.bar_plot(shap_values[0], feature_names=feature_names, show=False)
    st.pyplot(fig)

    st.caption("Positive values push toward approval; negative values push toward rejection.")

st.divider()
st.caption("Built with Scikit-learn, XGBoost, SHAP & Streamlit | Model trained on loan application data")
