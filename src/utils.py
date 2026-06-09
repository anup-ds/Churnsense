# src/utils.py
import pandas as pd

def load_data(path='data/WA_Fn-UseC_Telco.csv') -> pd.DataFrame:
    """Loads the raw Telco customer churn dataset."""
    return pd.read_csv(path)

def get_feature_columns(df: pd.DataFrame):
    """
    Filters out target and ID columns, then returns 
    a tuple of (numerical_columns, categorical_columns).
    """
    X = df.drop(columns=['Churn', 'customerID'], errors='ignore')
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    return num_cols, cat_cols

def churn_rate(df: pd.DataFrame, churn_col='Churn') -> float:
    """Calculates the overall churn percentage rate of the DataFrame."""
    if df[churn_col].dtype == 'object':
        # If target hasn't been mapped to 0/1 yet
        return (df[churn_col] == 'Yes').mean() * 100
    return df[churn_col].mean() * 100