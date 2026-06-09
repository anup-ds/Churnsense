# app/pages/01_🚀_Predict_Single.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 1. Dynamically find the main 'churnsense' project directory root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# 2. Force Python to include it in its module search paths
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# STYLE
st.markdown("""
<style>
section[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #2C3E50, #4CA1AF);
    color: white;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2C3E50, #4CA1AF);
    color: white;
}

}

</style>
""", unsafe_allow_html=True)

from src.model import load_model_xgb, predict_single

st.set_page_config(page_title="ChurnSense — Single Inference", page_icon="🧍", layout="wide")

st.title(" 🧍‍♂️ Single Customer Risk Assessment")
st.markdown("---")

# # Caches the ML model globally to prevent reloading on every rerun
@st.cache_resource
def get_inference_pipeline():
    return load_model_xgb()

try:
    model = get_inference_pipeline()
except Exception as e:
    st.error(f"Failed to load model file. Verify it exists in the models/ folder. Error: {e}")
    st.stop()

# Interactive Form Setup
with st.form("single_customer_form"):
    st.subheader("📋 Customer Demographics & Profile Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen Status", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        Partner = st.selectbox("Has Partner?", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Account Tenure (Months)", min_value=0, max_value=72, value=12)
        
    with col2:
        PhoneService = st.selectbox("Has Phone Service?", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Phone Lines?", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Provider Type", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security Add-on?", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup Add-on?", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection Add-on?", ["Yes", "No", "No internet service"])

    with col3:
        TechSupport = st.selectbox("Tech Support Add-on?", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV Services?", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movie Services?", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract Structure", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing Active?", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Mode", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        MonthlyCharges = st.number_input("Monthly Charge Base ($)", min_value=18.0, max_value=120.0, value=65.0)
        TotalCharges = st.number_input("Accumulated Total Charges ($)", min_value=18.0, max_value=9000.0, value=780.0)

    # Submission Action Button
    submit_btn = st.form_submit_button("🔮 Compute Churn Risk Profile")

if submit_btn:
    # Bundle input features into a payload dictionary matching model parameters
    customer_payload = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner, 'Dependents': Dependents,
        'tenure': tenure, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
        'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges
    }
    
    # Run backend inference
    result = predict_single(model, customer_payload)
    
    st.write("---")
    st.subheader("📊 Prediction Assessment Outputs")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Calculated Churn Probability", value=f"{result['churn_probability'] * 100:.2f}%")
        
    with res_col2:
        risk = result['risk_level']
        if risk == "HIGH":
            st.error(f"⚠️ **Risk Categorization: HIGH RISK PROFILE**")
        elif risk == "MEDIUM":
            st.warning(f"⚡ **Risk Categorization: MEDIUM RISK PROFILE**")
        else:
            st.success(f"✅ **Risk Categorization: LOW RISK PROFILE**")

    # ====================================================================
    #  MODEL EVALUATION VISUALIZATIONS
    # ====================================================================

    st.write("---")
    st.subheader("🎯 Model Performance & Explainability Analytics")

    tab3, = st.tabs(["🧬 SHAP Feature Importance"])

    with tab3:
        st.markdown("#### Customer-Specific SHAP Impact Explainer")
        st.write("Features pushing this specific customer toward or away from churning.")

        
        try:
            df_input = pd.DataFrame([customer_payload])

            raw_model = model.steps[-1][1]
            current_features = df_input.copy()
            feature_names = None

            for name, step in model.steps[:-1]:
                if "sampler" in name.lower() or "smote" in name.lower():
                    continue

                if hasattr(step, "transform"):
                    current_features = step.transform(current_features)
                    if hasattr(step, "get_feature_names_out"):
                        feature_names = step.get_feature_names_out()

            if hasattr(current_features, "toarray"):
                X_encoded = current_features.toarray()
            else:
                X_encoded = np.array(current_features)

            if feature_names is not None and X_encoded.shape[1] == len(feature_names):
                X_encoded = pd.DataFrame(X_encoded, columns=feature_names)
            elif hasattr(raw_model, "feature_names_in_"):
                X_encoded = pd.DataFrame(X_encoded, columns=raw_model.feature_names_in_)

           

            explainer = shap.TreeExplainer(raw_model)
            shap_values = explainer(X_encoded)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#1E293B')
            ax.set_facecolor('#1E293B')

            shap.plots.bar(shap_values[0], max_display=20, show=False)

            ax.tick_params(colors='white', labelsize=10)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for text in ax.texts:
                text.set_color('white')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Pipeline SHAP Error: {str(e)}")
            st.info("If your pipeline uses complex custom transformers, double check your notebook encoding steps.")