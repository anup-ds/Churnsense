# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

def get_feature_columns(df: pd.DataFrame):
    """
    Splits the dataframe into numerical and categorical feature groups,
    excluding target and unique identifier columns.
    """
    x  = df.drop(columns=['Churn', 'customerID'], errors='ignore')
    
    numeric_cols = x.select_dtypes(include = 'number').drop(columns = 'TotalCharges').columns.to_list()
    skewed_cols = x[['TotalCharges']].columns.to_list()
    categorical_cols = x.select_dtypes(include = 'object').drop(columns = 'Contract').columns.to_list()
    ordinal_cat_col = x[['Contract']].columns.to_list()

    # Isolate remaining categorical features (excluding 'Contract' if handled ordinally)
    #cat_cols = [col for col in X.select_dtypes(include=['object']).columns.tolist() if col != 'Contract']
    
    return numeric_cols, skewed_cols, categorical_cols, ordinal_cat_col

def build_preprocessor(numeric_cols, skewed_cols, categorical_cols, ordinal_cat_col):
    """
    Builds a structured ColumnTransformer matching the exact encoding
    and scaling specifications used during the experimentation phase.
    """
    #ordinal_cols = ['Contract']
    return ColumnTransformer(transformers=[
    ('skewed',  PowerTransformer(method='yeo-johnson'), skewed_cols),
    ('num',    StandardScaler(),  numeric_cols),
    ('cat',    OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('ordinal_cat',    OrdinalEncoder(categories=[['Month-to-month', 'One year', 'Two year']],
                                      handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cat_col)
                        ], remainder='drop')
    
    #return ColumnTransformer(transformers=[
       # ('num', StandardScaler(), num_cols),
       # ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
       # ('ordinal_cat', OrdinalEncoder(categories=[['Month-to-month', 'One year', 'Two year']],
         #                              handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols) ])