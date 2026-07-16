# app/pages/02_📦_Predict_Batch.py
import streamlit as st
import pandas as pd
from src.model import load_model_xgb, predict_batch

# STYLE
st.markdown("""
<style>
section[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #2C3E50, #4CA1AF);
    color: white;}
            
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2C3E50, #4CA1AF);
    color: white;}

</style>
""", unsafe_allow_html=True)


st.title("📅 High-Throughput Batch Inference Engine")
st.markdown("---")

# Load and cache model pipeline
@st.cache_resource
def get_inference_pipeline():
    return load_model_xgb()

try:
    model = get_inference_pipeline()
except Exception as e:
    st.error(f"Failed to load model file. Error: {e}")
    st.stop()

st.subheader("📤 Upload Customer Dataset")
st.markdown("Upload a raw `.csv` spreadsheet containing customer records. The engine will automatically evaluate risk parameters for every account.")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        #  Read input dataset
        input_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(input_df)} customer rows.")
        
        #  Run backend batch prediction
        with st.spinner("Processing batch calculations..."):
            predictions_df = predict_batch(model, input_df)

        st.success(f"🎉 Batch processing complete! Evaluated {len(predictions_df)} customer records.")

        #  Save the dataframe to session state so the analytics page can read it
        st.session_state['batch_predictions'] = predictions_df
        st.session_state['trained_model'] = model

        st.write("---")
        st.subheader("📊 Batch Inference Summary Preview")
        
        # Display a visual breakdown of the high/medium/low risk flags generated
        risk_counts = predictions_df['risk_level'].value_counts()
        
        count_col1, count_col2, count_col3 = st.columns(3)
        with count_col1:
            st.metric("🚨 High Risk Accounts Identified", risk_counts.get("HIGH", 0))
        with count_col2:
            st.metric("⚡ Medium Risk Accounts Identified", risk_counts.get("MEDIUM", 0))
        with count_col3:
            st.metric("✅ Low Risk Accounts Identified", risk_counts.get("LOW", 0))
            
        st.write("---")
        st.markdown("### 📋 Processed Data Preview")
        
        # Display the output table focusing on identifying keys and newly added column
        preview_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges','TotalCharges', 'churn_probability', 'risk_level']
        existing_preview_cols = [col for col in preview_cols if col in predictions_df.columns]
        
        st.dataframe(predictions_df[existing_preview_cols].head(10), use_container_width=True)
        
        # 3. Generate a clean CSV string output for downloading
        gen_csv = predictions_df.to_csv(index=False).encode('utf-8')
        
        st.write("---")
        st.download_button(
            label="📥 Download Annotated Batch Predictions CSV",
            data=gen_csv,
            file_name="churnsense_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error compiling batch spreadsheet processing. Ensure data schema matches features. Details: {e}")
        
        st.markdown("---")
        st.subheader("📊 Explore Batch Diagnostics")
        st.markdown("Ready to view deeper statistical breakdowns, distributions, and churn correlations for this specific batch?")

# 2. Render the primary navigation button
    if st.button("📊 Go to Batch Analytics", type="primary"):
        st.switch_page("pages/03_📊_Analytics.py")