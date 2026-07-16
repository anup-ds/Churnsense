# app/streamlit_app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from src.model import load_model_xgb
import shap
import matplotlib.pyplot as plt
import numpy as np

# 🚀 Datively find the root project directory and append it to Python's search path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if root_path not in sys.path:
    sys.path.insert(0, root_path)


# 1. Page Configuration
st.set_page_config(
    page_title="ChurnSense — Analytics Hub",
    page_icon= "https://miro.medium.com/0*dzmm3qresODlScte",
    layout="wide"
)


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

</style>
""", unsafe_allow_html=True)

# 2. Cached Data Loader
@st.cache_data
def load_data():
    """Loads and caches dataset to maintain quick interface rendering."""
    data_path = os.path.join(root_path, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        
        # ✨ FIX THE VALUE ERROR HERE:
        # Convert empty strings " " to NaN, then fill them with 0
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'] = data['TotalCharges'].fillna(0)
        
        return data
    # Fallback structure if the dataset is moved
    return pd.DataFrame({'Churn': ['No']*73 + ['Yes']*27})

df = load_data()

# 3. App Title Header
st.title("ChurnSense — Customer Analytics Hub")
st.markdown("### *AI-Powered Customer Churn Prediction Framework*")
st.write("---")
img1, img2, img3 = st.columns([3, 4, 3])

with img2: # This places the image inside the center column
    st.image(
        "https://miro.medium.com/0*dzmm3qresODlScte", 
        caption="Analytics Overview",
        width=300
    )
st.write("---")
# 4. KPI Summary Cards Row
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
# Calculate baseline churn directly from data
baseline_churn = (df['Churn'].isin(['Yes', 1])).mean() * 100

with col1:
    st.metric(label="Total Monitored Accounts", value=f"{total_customers:,}")

with col2:
    st.metric(
        label="Observed Churn Rate", 
        value=f"{baseline_churn:.1f}%",
        help="The percentage of customers that have historically churned based on the dataset."
    )

with col3:
    st.metric(label="Risk Detection Rate (Recall)", 
        value="87.6%", 
        help="The percentage of leaving customers that the AI successfully catches before they walk out."
    )



with col4:
    st.metric(
        label="Model ROC-AUC Score", 
        value="84.5%", 
        help="Overall capability of the model to distinguish between churning and loyal customers."
    )

st.write("---")
# 5. Descriptive Insights Grid
left_col, right_col = st.columns(2)
with left_col:
    st.subheader("📊 Operational Objectives")
    st.markdown("""
    * **Early Risk Identification:** Target high-risk consumer profiles before contract expirations.
    * **Proactive Retention:** Optimize marketing outreach allocation using precise data.
    * **High-Throughput Analytics:** Streamline batch assessment operations via raw spreadsheet processing.
    """)
    
with right_col:
 
    # 1. Load the cached model pipeline
    @st.cache_resource
    def get_inference_pipeline():
        return load_model_xgb()
 
    try:
        model = get_inference_pipeline()
    except Exception as e:
        st.error(f"Failed to load model file. Error: {e}")
        st.stop()
 
    # 2. Extract Fitted Preprocessor from the Loaded Model Pipeline
    try:
        prep = model.named_steps['preprocessor']
        raw_classifier = model.named_steps['model']
    except Exception as e:
        st.error(f"Failed to isolate pipeline steps. Error: {e}")
        st.stop()
 
    # 3. Transform data for SHAP evaluation
    try:
        # Drop target variable to match feature spaces
        X_test = df.drop(columns=['Churn'], errors='ignore')
        X_test_transformed = prep.transform(X_test)
        
        # Extract and clean feature names
        feature_names = prep.get_feature_names_out().tolist()
        clean_names = [name.split('__', 1)[-1].replace('_', ' ') for name in feature_names]
    except Exception as e:
        st.error(f"Failed data preparation for SHAP. Error: {e}")
        st.stop()
 
    # 4. Compute Tree SHAP Values and Plot
    try:
        st.subheader("🔬 Feature Importance Profile")
        st.caption("Bars show average magnitude of impact on churn risk (not direction). See **Analytics** for the full breakdown and signed effects.")
        
        # Calculate summary feature weights
        explainer = shap.TreeExplainer(raw_classifier)
        shap_values = explainer(X_test_transformed)
        
        # Inject pristine feature names into the SHAP object before drawing
        shap_values.feature_names = clean_names
 
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('none')      # Transparent outer background
        ax.set_facecolor('#1E293B')
 
        # Homepage teaser: top 8 features only. Full 15+ feature view lives on Analytics page.
        shap.plots.bar(shap_values, max_display=8, show=False)
 
        for patch in ax.patches:
            patch.set_facecolor('#06B6D4')  # Elegant Cyan bar fill
            patch.set_edgecolor('#FFFFFF')  # Crisp White bar border
            patch.set_alpha(0.9)
 
        for text in ax.texts:
            text.set_color('#FFFFFF')      # Forces all bar labels to white
            text.set_fontsize(10)
            # mean(|SHAP value|) is a magnitude, not a signed contribution —
            # strip the misleading "+" prefix that implies "increases churn"
            label = text.get_text().strip()
            if label.startswith('+'):
                text.set_text(label[1:])
 
        # Style the feature name labels along the Y-axis
        ax.tick_params(axis='y', colors='#F8FAFC', labelsize=12) 
        
        # Style the numerical impact value labels along the X-axis
        ax.tick_params(axis='x', colors="#ECF3F7", labelsize=11) # Sky blue axis metrics
        ax.xaxis.label.set_color("#E1E9ED")
 
        # Change the bounding frame/spines color
        for spine in ax.spines.values():
            spine.set_color('#0284C7')      # Ocean Blue borders
            spine.set_linewidth(1.5)
  
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Could not render SHAP visualization. Error details: {e}")
