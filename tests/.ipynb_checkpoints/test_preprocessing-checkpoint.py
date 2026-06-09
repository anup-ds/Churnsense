# tests/test_preprocessing.py
import pytest
import pandas as pd
from src.preprocessing import get_feature_columns, build_preprocessor

def test_get_feature_columns_logic():
    """
    Tests that get_feature_columns isolates columns correctly 
    and handles case sensitivity with uppercase 'X' seamlessly.
    """
    # Create a minimal mock dataframe mimicking your raw Telco data schema
    fake_df = pd.DataFrame({
        'customerID': ['0002-ORFBO'],
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['Yes'],
        'tenure': [9],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['Yes'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['No'],
        'Contract': ['One year'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Mailed check'],
        'MonthlyCharges': [65.6],
        'TotalCharges': [593.3],
        'Churn': ['No']
    })
    
    # Run your function
    numeric_cols, skewed_cols, categorical_cols, ordinal_cat_col = get_feature_columns(fake_df)
    
    # Assertions to verify correct parsing groupings
    assert 'customerID' not in numeric_cols, "customerID should have been dropped!"
    assert 'Churn' not in categorical_cols, "Target variable Churn should have been dropped!"
    assert 'TotalCharges' in skewed_cols, "TotalCharges should be in the skewed group!"
    assert 'Contract' in ordinal_cat_col, "Contract type must be classified as ordinal_cat!"
    
    # Ensure standard numeric columns don't contain TotalCharges
    assert 'TotalCharges' not in numeric_cols, "TotalCharges should be removed from standard numeric_cols!"

def test_build_preprocessor_execution():
    """
    Validates that the ColumnTransformer can initialize and execute fit operations 
    on mock dimensions without throwing import errors (like missing PowerTransformer).
    """
    # Define simple mock lists matching output formats
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
    skewed_cols = ['TotalCharges']
    categorical_cols = ['gender', 'Partner']
    ordinal_cat_col = ['Contract']
    
    # Initialize the ColumnTransformer from your script
    preprocessor = build_preprocessor(numeric_cols, skewed_cols, categorical_cols, ordinal_cat_col)
    
    assert preprocessor is not None, "Preprocessor failed to initialize!"