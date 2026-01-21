import numpy as np
import plotly.graph_objects as go
from src.brain.state_estimator import StateEstimator

def test_kalman_filter():
    # 1. Generate synthetic "True Demand" (Random Walk)
    n_days = 100
    true_demand = np.zeros(n_days)
    true_demand[0] = 50
    for i in range(1, n_days):
        # Demand slowly drifts
        true_demand[i] = true_demand[i-1] + np.random.normal(0, 0.5) 

    # 2. Generate "Observed Sales" (Noisy Data)
    # Poisson-ish noise, but approximated with Normal for simplicity here
    observed_sales = true_demand + np.random.normal(0, 10, n_days)
    
    # 3. Apply Kalman Filter
    estimator = StateEstimator(process_noise=0.1, measurement_noise=100.0) # High measurement noise
    estimated_demand, uncertainties = estimator.run_filter(observed_sales)
    
    # 4. Visualization Logic (Console Output)
    print("Step | True Demand | Observed | Estimated | Uncert (P)")
    print("-" * 60)
    for i in range(0, n_days, 10): # Print every 10th step
        print(f"{i:4d} | {true_demand[i]:10.2f}  | {observed_sales[i]:8.2f} | {estimated_demand[i]:9.2f} | {uncertainties[i]:10.2f}")
        
    mse = np.mean((true_demand - estimated_demand) ** 2)
    raw_mse = np.mean((true_demand - observed_sales) ** 2)
    
    print(f"\nRaw Data MSE: {raw_mse:.2f}")
    print(f"Kalman Filter MSE: {mse:.2f}")
    
    if mse < raw_mse:
        print("\nSUCCESS: Kalman Filter successfully reduced noise!")
    else:
        print("\nWARNING: Kalman Filter did not improve signal quality. Tuning needed.")

if __name__ == '__main__':
    test_kalman_filter()
