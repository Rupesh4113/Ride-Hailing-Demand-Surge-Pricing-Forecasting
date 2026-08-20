import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import get_data
from features import preprocess_and_aggregate, engineer_features
from model import load_data, train_models
from surge import apply_surge_pricing_to_predictions

def main():
    print("====================================================")
    print("STARTING RIDE-HAILING FORECASTING PIPELINE")
    print("====================================================\n")
    
    # Step 1: Get Data
    df_raw = get_data()
    
    # Step 2: Preprocess & Aggregate & Feature Engineering
    grid, active_h3 = preprocess_and_aggregate(df_raw)
    features_df = engineer_features(grid)
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    features_path = os.path.join(data_dir, "features.parquet")
    features_df.to_parquet(features_path, index=False)
    print(f"Features saved to {features_path}\n")
    
    # Step 3: Train and Evaluate Models
    lgb_model, xgb_model, test_df, feature_cols = train_models(features_df)
    
    # Step 4: Make Predictions on Test Set (using LightGBM)
    print("\nGenerating forecasts using trained LightGBM model...")
    X_test = test_df[feature_cols]
    test_df['predicted_demand'] = lgb_model.predict(X_test)
    
    # Ensure predicted demand is non-negative
    test_df['predicted_demand'] = np.maximum(0, test_df['predicted_demand'])
    
    # Step 5: Apply Surge Pricing Multipliers
    print("Simulating surge pricing multipliers...")
    predictions_with_surge = apply_surge_pricing_to_predictions(
        test_df, 
        demand_col='predicted_demand', 
        supply_col='supply_lag_1'
    )
    
    # Save final predictions
    predictions_path = os.path.join(data_dir, "predictions_with_surge.parquet")
    predictions_with_surge.to_parquet(predictions_path, index=False)
    
    print(f"\nSaved predictions with surge pricing to {predictions_path}")
    print("\n====================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("====================================================")

if __name__ == "__main__":
    main()
