# app/pages/03_📊_Analytics.py
from unicodedata import name

import matplotlib
from sklearn.metrics import auc, roc_curve
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

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



st.title("📊 Batch Analytics Dashboard")
st.markdown("---")

# CHECK IF BATCH DATA EXISTS IN SESSION STATE
if 'batch_predictions' not in st.session_state:
    st.warning("⚠️ No active batch data found! Please upload a customer dataset on the Batch Prediction page first to view diagnostics.")
    
    # Navigation back to the upload screen
    if st.button("⬅️ Go to Batch Prediction Page", type="primary"):
        st.switch_page("pages/02_📅_Predict_Batch.py")
        
    st.stop()

# IF DATA EXISTS, LOAD IT
df = st.session_state['batch_predictions']

# Layout splits for visual cards
kp1, kp2, kp3 = st.columns(3)
total_cust = len(df)

# Check how your predict_batch handles the churn flag (adjust strings/numbers to match your output columns)
# This assumes you have 'risk_level' and 'churn_probability' columns from your tests
high_risk_count = len(df[df['risk_level'] == 'HIGH'])
avg_prob = df['churn_probability'].mean()

with kp1:
    st.metric("Batch Dataset Size", f"{total_cust} Accounts")
with kp2:
    st.metric("High Churn Risk Customers", f"{high_risk_count} Accounts", delta=f"{(high_risk_count/total_cust)*100:.1f}% Total")
with kp3:
    st.metric("Average Churn Probability", f"{avg_prob:.1%}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Risk Profile Segmentation")
    # Pie chart breaking down risk bands (LOW, MEDIUM, HIGH)
    fig_pie = px.pie(
        df, 
        names='risk_level', 
        title='Proportion of Customers by Business Risk Level',
        color='risk_level',
        color_discrete_map={'LOW': '#2ec4b6', 'MEDIUM': '#ff9f1c', 'HIGH': '#e71d36'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("📈 Probability Distribution Profile")
    # Histogram showcasing the velocity/spread of calculated mathematical decimals
    fig_hist = px.histogram(
        df, 
        x='churn_probability', 
        nbins=25, 
        title='Distribution Grid of Customer Churn Probabilities',
        labels={'churn_probability': 'Calculated Churn Probability'},
        color_discrete_sequence=["#3ede1e"]
    )
    fig_hist.update_layout(yaxis_title_text='Customer Count')
    st.plotly_chart(fig_hist, use_container_width=True)

tab1, tab2, tab3, tab4= st.tabs(["Confusion Matrix", "Roc-Auc Curve", "Precision-Recall Curve", "SHAP Feature Importance"])

   # Dynamic guard check to see if ground-truth target is provided
#has_true_labels = 'Churn' in df.columns
with tab1:
    # --- ADD THIS: PLOTLY CONFUSION MATRIX IN TAB 1 ---
        st.markdown("---")
        st.subheader("🧩 Confusion Matrix")
        st.write("Track true classification boundaries against actual false alarms.")
        
        from sklearn.metrics import confusion_matrix
        import plotly.express as px
        y_true = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' or x == 1 else 0)

        # Compute raw confusion matrix values
        # Assumes your model threshold defaults to 0.5 for binary classification
        # Optimize the binary threshold to account for the 2.77x minority class weight scale
        prediction_threshold = 0.38 
        y_pred = (df['churn_probability'] > prediction_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Build an interactive Plotly Heatmap
        cm_df = pd.DataFrame(cm, index=['Actual No Churn', 'Actual Churn'], columns=['Predicted No Churn', 'Predicted Churn'])
        
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            labels=dict(x="Model Prediction", y="Ground Truth", color="Customer Count")
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

with tab2:
    st.subheader("🔍 ROC-AUC Curve Analysis")
    st.write("Visualize the trade-off between True Positive Rate and False Positive Rate across different thresholds.")
    
    #if has_true_labels:
    from sklearn.metrics import roc_curve, auc
        
        # Normalize target string indicators to binary integers (0 and 1)
    y_true = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' or x == 1 else 0)
    y_prob = df['churn_probability']
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
        
        # Build clean Matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('none')  # Transparent canvas background
    ax.set_facecolor('#1e293b')      # Slate dark plot background
        
    ax.plot(fpr, tpr, color='#ff9f1c', label=f'(Score= {roc_auc:.4f})', linewidth=2.5)
    ax.plot([0, 1], [0, 1], color='white', linestyle='--', alpha=0.5)
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('Receiver Operating Characteristic (ROC)', color='white', pad=10)
    ax.tick_params(colors='white')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=':', alpha=0.3)
        
        # Render the canvas directly onto the Streamlit view layer
    st.pyplot(fig)
##       st.info("💡 **Production Matrix Guard:** ROC-AUC calculations require a ground-truth classification target. The uploaded batch dataset does not contain a raw `'Churn'` column to evaluate historical error metrics against.")


with tab3:
    st.subheader("📊 Precision-Recall Curve Analysis")
    st.write("Evaluate the balance between precision and recall for different classification thresholds.")
    
    
    #if has_true_labels:
    from sklearn.metrics import precision_recall_curve, average_precision_score
        
    y_true = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' or x == 1 else 0)
    y_prob = df['churn_probability']
        
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
        
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#1e293b')
        
    ax.plot(recall, precision, color='#2ec4b6', linewidth=2.5)
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', color='white')
    ax.set_ylabel('Precision', color='white')
    ax.set_title('Precision-Recall Curve', color='white', pad=10)
    ax.tick_params(colors='white')
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=':', alpha=0.3)
        
    st.pyplot(fig)
    #else:
       # st.info("💡 **Production Matrix Guard:** Precision-Recall curve visualizations require a ground-truth target. Upload a historical evaluation dataset containing a verified `'Churn'` column to map active model recall precision.")
    # Placeholder for Precision-Recall curve (requires true labels and predicted probabilities)

with tab4:
    st.subheader("🔬 SHAP Feature Importance Analysis")
    st.write("Understand which features are driving the model's predictions the most.")

    # Check if the model was passed down successfully in session state
    if 'trained_model' not in st.session_state:
        st.error("⚠️ Model object pipeline not found in session state. Please re-run the Batch Prediction upload first.")
    else:
        import matplotlib.pyplot as plt
        import numpy as np

            # 1. Retrieve the pipeline object from session state
        pipeline_model = st.session_state['trained_model']

            # 2. Extract the preprocessor and classifier steps from the Pipeline
        preprocessor = pipeline_model.named_steps['preprocessor']
        xgb_classifier = pipeline_model.named_steps['model']

            # Helper function to clean feature names
        def clean_feature_name(name):
                clean_name = (name.replace('cat: ', '')
                          .replace('ordinal cat: ', '')
                          .replace('cat__', '')
                          .replace('ordinal_cat__', '')
                          .replace('num__', '')
                          .replace('ordinal__', '')
                          .replace('ordinal', '')
                          .replace('skewed__', '')
                          .replace('num:', ''))
        
                # Handle one-hot encoded categories: "InternetService__Fiber optic" → "Internet Service: Fiber optic"
                if '__' in clean_name:
                    parts = clean_name.split('__', 1)
                    if len(parts) == 2:
                        column_name, category_value = parts
                        return f"{column_name.replace('_', ' ')}: {category_value}"
        
                return clean_name.replace('_', ' ')

            # 3. Process data & compute True SHAP values
        try:
                import shap
        
                # Transform the batch data currently stored in session state
                df_batch = df
                X_batch = df_batch.drop(columns=['Churn', 'customerID'], errors='ignore')
                X_transformed = preprocessor.transform(X_batch)
        
                # Get cleaned feature names
                raw_names = preprocessor.get_feature_names_out().tolist()
                cleaned_names = [clean_feature_name(name) for name in raw_names]
        
                # Calculate SHAP explanation values
                explainer = shap.TreeExplainer(xgb_classifier)
                shap_values = explainer(X_transformed)
                shap_values.feature_names = cleaned_names
        
        except Exception as e:
                st.error(f"Failed to calculate global SHAP values. Error: {e}")
                st.stop()

    # 4. Render the styled matplotlib SHAP chart
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('#1e293b')

    # Draw the true global SHAP summary bar plot
    shap.plots.bar(shap_values, max_display=20, show=False)

    # Style overrides: Change bars to cyan and text labels to white
    for patch in ax.patches:
        patch.set_facecolor('#06B6D4')  # Cyan bar fill
        patch.set_edgecolor('#FFFFFF')  # White border
        patch.set_alpha(0.9)

    for text in ax.texts:
        text.set_color('#FFFFFF')      # White numeric labels (+0.368, etc.)
        text.set_fontsize(9)

    ax.set_title('Global SHAP Feature Importance Profile', color='white', fontsize=12, pad=15)
    ax.tick_params(colors='white', labelsize=9)
    ax.xaxis.label.set_color('white')
    
    for spine in ax.spines.values():
        spine.set_color('#0284C7')

    st.pyplot(fig)







































































































            # 1. Retrieve the pipeline object from session state
     #       pipeline_model = st.session_state['trained_model']
            
            # 2. Extract the preprocessor and classifier steps from the Pipeline
       #     preprocessor = pipeline_model.named_steps['preprocessor']
      #      xgb_classifier = pipeline_model.named_steps['model']
            

            #feature_names = preprocessor.get_feature_names_out().tolist()
            #feature_names = [name.split('__', 1)[-1] for name in feature_names]

            #encoded_feature_names = feature_names
            # Helper function to clean feature names
       #     def clean_feature_name(name):
            #"""
            #Strip preprocessing prefixes and format for readability.
           # E.g., "cat: InternetService__Fiber optic" → "Internet Service: Fiber optic"
           # """
    # Strip preprocessing prefixes (cat:, ordinal cat:, etc.)
       #         clean_name = name.replace('cat: ', '').replace('ordinal cat: ', '').replace('cat__', '').replace('ordinal_cat__', '').replace('num', '').replace('ordinal', '').replace('skewed', '')
    # Handle one-hot encoded categorical: "InternetService__Fiber optic" → "Internet Service: Fiber optic"
                
        ##        if '__' in clean_name:
        #            parts = clean_name.split('__', 1)
        #            if len(parts) == 2:
        #                column_name, category_value = parts
        #            return f"{column_name.replace('_', ' ')}: {category_value}"
    
    # For ordinal and other features: just replace underscores with spaces
        #        return clean_name.replace('_', ' ')


        #    feature_names = preprocessor.get_feature_names_out().tolist()
         #   feature_names = [clean_feature_name(name) for name in feature_names]

        #    encoded_feature_names = feature_names

           #st.write(feature_names)

        

            # 4. Extract mathematical feature importances directly from the booster
        #    importances = xgb_classifier.feature_importances_

            # 5. Map, sort, and isolate the top 10 most influential features
        #    feature_importance_map = sorted(
         #       zip(encoded_feature_names, importances), 
         #       key=lambda x: x[1], 
        #        reverse=True)[:20]
            
         #   top_features, top_weights = zip(*feature_importance_map)
         #   y_pos = np.arange(len(top_features))

            # 6. Render the dark-themed matplotlib chart canvas
         #   fig, ax = plt.subplots(figsize=(10, 8))
         #   fig.patch.set_facecolor('none')
         #   ax.set_facecolor('#1e293b')

         #   bars = ax.barh(y_pos, top_weights, align='center', color='#ff007f', alpha=0.9, edgecolor='white', height=0.6)
          ##  ax.bar_label(bars, fmt=' %.3f', color='white', fontsize=9, padding=4)
         #   ax.set_yticks(y_pos)
         #   ax.set_yticklabels(top_features, color='white', fontsize=9)
         #   ax.invert_yaxis()  # Most critical drivers on top
            
        #    ax.set_xlabel('Relative Gain Importance (XGBoost Feature Weight)', color='white', fontsize=9)
         #   ax.set_title('Live Pipeline Feature Importance Profile', color='white', fontsize=11, pad=10)
         #   ax.tick_params(colors='white', labelsize=9)
          #  ax.grid(True, axis='x', linestyle=':', alpha=0.3)

         #   for spine in ['top', 'right', 'bottom', 'left']:
         #       ax.spines[spine].set_visible(False)

         #   st.pyplot(fig)

     #   except Exception as e:
      #      st.error(f"Could not extract features dynamically from the model object pipeline. Details: {e}")


# Add a floating shortcut back to the uploader at the bottom for great UX
st.markdown("---")
if st.button("🔄 Upload Another Batch File"):
    st.switch_page("pages/02_📅_Predict_Batch.py")