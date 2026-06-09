# src/model.py
import os
import joblib
import pandas as pd

# --- THIS PATTERN PATCHES THE SCYLLA/SKLEARN VERSION DEVIATION ---
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    # Dynamically inject the fallback attribute to keep joblib.load happy
    class DummyRemainder:
        def __init__(self, *args, **kwargs): pass
    sklearn.compose._column_transformer._RemainderColsList = DummyRemainder
# -----------------------------------------------------------------

def load_model(path=None):
    """
    Loads the trained scikit-learn Pipeline from the specified path.
    Uses robust relative routing to prevent FileNotFoundError bugs across different environments.
    """
    if path is None:
        # Resolves path dynamically relative to where this script file lives
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, '..', 'models', 'churn_model_v1.pkl')
        
    return joblib.load(path)

def predict_single(model, customer: dict) -> dict:
    """
    Predicts the churn profile for a single customer transaction payload.
    Returns a clean dictionary mapped to business-friendly risk categories.
    """
    # Wrap the input data dictionary into a DataFrame structure required by the pipeline
    df = pd.DataFrame([customer])
    
    # Generate the probability array for the positive class (Churned)
    prob = model.predict_proba(df)[0][1]
    
    # Map raw math decimals into concrete risk profile bands
    if prob > 0.7:
        risk_level = 'HIGH'
    elif prob > 0.4:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
        
    return {
        'churn_probability': round(float(prob), 4),
        'will_churn': bool(prob > 0.5),
        'risk_level': risk_level
    }

def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes high-throughput batch inference across an entire customer DataFrame.
    Appends risk calculations directly to a fresh copy of the input dataset.
    """
    # Working on a duplicate copy prevents unexpected data mutating side effects
    results_df = df.copy()
    
    # Generate an explicit array of prediction decimals for all rows
    probabilities = model.predict_proba(df)[:, 1]
    
    # Standardize predictions to a binary boolean mask based on standard default thresholding
    predictions = probabilities > 0.5
    
    # Assign new feature series transformations to the frame output
    results_df['churn_probability'] = probabilities.round(4)
    results_df['will_churn'] = predictions.astype(int)
    results_df['risk_level'] = results_df['churn_probability'].apply(
        lambda p: 'HIGH' if p > 0.7 else ('MEDIUM' if p > 0.4 else 'LOW')
    )
    
    return results_df