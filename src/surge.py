import numpy as np
import pandas as pd

def calculate_surge_multiplier(predicted_demand, estimated_supply, alpha=0.6, epsilon=2.0, max_surge=3.5):
    """
    Computes a surge pricing multiplier based on demand-supply imbalance.
    
    Formula:
        Surge = 1.0 + alpha * max(0, predicted_demand - estimated_supply) / (estimated_supply + epsilon)
    
    Parameters:
    - predicted_demand: Series or array of predicted pickup counts
    - estimated_supply: Series or array of estimated driver supply (e.g. lagged dropoffs)
    - alpha: Multiplier sensitivity factor
    - epsilon: Smoothing parameter to avoid division by zero and dampen surges for very low volumes
    - max_surge: Cap on the maximum surge multiplier (e.g. 3.5x)
    """
    # Ensure arrays
    d = np.maximum(0, np.array(predicted_demand))
    s = np.maximum(0, np.array(estimated_supply))
    
    imbalance = d - s
    
    # Calculate multipliers
    multipliers = 1.0 + alpha * (imbalance) / (s + epsilon)
    
    # Only apply surge where demand > supply
    multipliers = np.where(imbalance > 0, multipliers, 1.0)
    
    # Clip to maximum surge limit
    multipliers = np.clip(multipliers, 1.0, max_surge)
    
    return np.round(multipliers, 2)

def apply_surge_pricing_to_predictions(predictions_df, demand_col='predicted_demand', supply_col='supply_lag_1'):
    """
    Adds surge multipliers to the predictions dataframe.
    """
    df = predictions_df.copy()
    df['surge_multiplier'] = calculate_surge_multiplier(df[demand_col], df[supply_col])
    return df

if __name__ == "__main__":
    # Test cases
    test_demands = [0.5, 1.0, 5.0, 10.0, 20.0, 15.0, 2.0]
    test_supplies = [0.0, 2.0, 1.0,  5.0,  2.0, 20.0, 0.0]
    
    results = pd.DataFrame({
        'Demand': test_demands,
        'Supply': test_supplies
    })
    results['Surge'] = calculate_surge_multiplier(results['Demand'], results['Supply'])
    print("Surge Pricing Test Outputs:")
    print(results.to_string(index=False))
