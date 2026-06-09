import pytest
import pandas as pd
import sys
# Add these definitions at the bottom of your test file to verify model.py
from src.model import load_model_xgb, predict_single, predict_batch

"""@pytest.fixture
def trained_pipeline():
    ""Fixture that automatically fetches your serialized model checkpoint.""
    return load_model()"""
@pytest.fixture
def model():
    """Fixture to load the trained model checkpoint."""
    return load_model_xgb()

@pytest.fixture
def mock_customer_payload():
    """Standard customer entry footprint to test transactional payloads."""
    return {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 12, 'PhoneService': 'Yes', 'MultipleLines': 'No', 
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes', 
        'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 80.0, 'TotalCharges': 960.0
    }

def test_model_loads(model):
    """Test 1: Verifies the model unpickles and loads without throwing errors."""
    assert model is not None ,"Model failed to load"

def test_single_inference_bounds(model, mock_customer_payload):
    """Ensures individual predictions output clean data types and bounded probabilities."""
    res = predict_single(model, mock_customer_payload)
    assert 0.0 <= res['churn_probability'] <= 1.0
    assert isinstance(res['will_churn'], bool)
    assert res['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']

def test_batch_inference_dimensions(model, mock_customer_payload):
    """Validates batch matrix transformations maintain structural data limits."""
    # Convert dummy data dict to a multi-row testing table framework
    batch_df = pd.DataFrame([mock_customer_payload] * 3)
    output_df = predict_batch(model, batch_df)
    
    # Confirm output rows map to structural entries exactly
    assert len(output_df) == 3
    assert 'churn_probability' in output_df.columns
    assert 'risk_level' in output_df.columns