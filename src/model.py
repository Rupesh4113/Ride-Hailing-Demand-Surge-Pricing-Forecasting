import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(features_path):
    """
    Loads features and prepares H3 category encoding.
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file {features_path} not found. Run features.py first.")
        
    df = pd.read_parquet(features_path)
    df['h3_index'] = df['h3_index'].astype('category')
    return df

def train_models(df, models_dir="models"):
    """
    Splits data chronologically and trains XGBoost and LightGBM models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Ensure h3_index is categorical type
    df = df.copy()
    df['h3_index'] = df['h3_index'].astype('category')
    
    # Chronological Split: Train on first 10 days, Test on remaining 4 days
    split_date = df['timestamp'].min() + pd.Timedelta(days=10)
    print(f"Splitting dataset chronologically at {split_date}")
    
    train = df[df['timestamp'] < split_date].copy()
    test = df[df['timestamp'] >= split_date].copy()
    
    features = [
        'h3_index', 'demand_lag_1', 'demand_lag_2', 'demand_lag_48', 'supply_lag_1',
        'demand_roll_mean_3', 'demand_roll_std_3', 'demand_roll_mean_6', 'demand_roll_std_6',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'spatial_demand_lag_1'
    ]
    target = 'target_demand'
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 1. Train LightGBM
    print("Training LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # 2. Train XGBoost
    print("Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        enable_categorical=True,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Save models
    lgb_path = os.path.join(models_dir, "lightgbm_model.pkl")
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    
    joblib.dump(lgb_model, lgb_path)
    joblib.dump(xgb_model, xgb_path)
    print(f"Saved models to {lgb_path} and {xgb_path}")
    
    # Evaluate
    evaluate_and_save(lgb_model, xgb_model, X_test, y_test, features)
    
    return lgb_model, xgb_model, test, features

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def evaluate_and_save(lgb_model, xgb_model, X_test, y_test, features, output_dir="data"):
    """
    Evaluates both models, prints metrics, and saves feature importances plot.
    """
    y_pred_lgb = lgb_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    
    metrics = []
    for name, y_pred in [("LightGBM", y_pred_lgb), ("XGBoost", y_pred_xgb)]:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        s_mape = smape(y_test, y_pred)
        
        metrics.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "sMAPE (%)": round(s_mape, 2)
        })
        
    metrics_df = pd.DataFrame(metrics)
    print("\n--- Evaluation Metrics on Test Set ---")
    print(metrics_df.to_string(index=False))
    
    # Save metrics report
    metrics_df.to_csv(os.path.join(output_dir, "metrics_report.csv"), index=False)
    
    # Save feature importances (using LightGBM)
    importance = lgb_model.feature_importances_
    # Exclude categories if needed, but plotting them is fine
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=150)
    plt.close()
    print(f"Saved feature importances plot to {os.path.join(output_dir, 'feature_importances.png')}")

if __name__ == "__main__":
    df = load_data("data/features.parquet")
    train_models(df)
