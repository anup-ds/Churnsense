# run_batch_check.py
import pandas as pd
from src.model import load_model, predict_batch

print("📦 Loading new customer batch data...")
new_data = pd.read_csv("data/sample_new_customers.csv")

print("🤖 Loading trained machine learning pipeline...")
model = load_model()

print("🔮 Running high-throughput batch inference...")
predictions_df = predict_batch(model, new_data)

print("\n📊 Verification Summary:")
print(f"Processed Rows: {len(predictions_df)}")
print("Output Columns added successfully:", [col for col in predictions_df.columns if col not in new_data.columns])

print("\n📋 Sample Results Preview:")
print(predictions_df[['tenure', 'Contract', 'churn_probability', 'risk_level']])