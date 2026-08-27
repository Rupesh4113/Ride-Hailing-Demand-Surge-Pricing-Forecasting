import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import get_data
from features import preprocess_and_aggregate, engineer_features
from model import train_models
from surge import apply_surge_pricing_to_predictions

def main():
    parser = argparse.ArgumentParser(description="Ride-Hailing Spatio-Temporal Demand Forecasting Pipeline")
    parser.add_argument(
        "--source", 
        type=str, 
        choices=["synthetic", "nyc_cloud"], 
        default="nyc_cloud",
        help="Data source: 'synthetic' for Chicago synthetic data or 'nyc_cloud' for NYC Green Taxi data."
    )
    args = parser.parse_args()
    
    source = args.source
    print("====================================================")
    print(f"STARTING RIDE-HAILING FORECASTING PIPELINE FOR {source.upper()}")
    print("====================================================\n")
    
    # Step 1: Get Data
    df_raw = get_data(source=source)
    
    # Step 2: Preprocess & Aggregate & Feature Engineering
    # Resolution 8 (area ~0.7 sq km)
    grid, active_h3 = preprocess_and_aggregate(df_raw, resolution=8)
    features_df = engineer_features(grid)
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    features_path = os.path.join(data_dir, f"features_{source}.parquet")
    features_df.to_parquet(features_path, index=False)
    print(f"Features saved to {features_path}\n")
    
    # Step 3: Train and Evaluate Models
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", source)
    lgb_model, xgb_model, test_df, feature_cols = train_models(features_df, models_dir=models_dir)
    
    # Make sure metrics file names are updated in model evaluation
    # (train_models calls evaluate_and_save internally, which writes to data/metrics_report.csv
    # We will adjust model.py to write to data/metrics_report_{source}.csv)
    # To do this cleanly, we can rename the generated files or update src/model.py
    
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
    predictions_path = os.path.join(data_dir, f"predictions_with_surge_{source}.parquet")
    predictions_with_surge.to_parquet(predictions_path, index=False)
    
    # Rename generated assets from model.py to match source
    if os.path.exists(os.path.join(data_dir, "metrics_report.csv")):
        os.rename(
            os.path.join(data_dir, "metrics_report.csv"),
            os.path.join(data_dir, f"metrics_report_{source}.csv")
        )
    if os.path.exists(os.path.join(data_dir, "feature_importances.png")):
        os.rename(
            os.path.join(data_dir, "feature_importances.png"),
            os.path.join(data_dir, f"feature_importances_{source}.png")
        )
    
    print(f"\nSaved predictions with surge pricing to {predictions_path}")
    print("\n====================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("====================================================")

if __name__ == "__main__":
    main()
