# scripts/generate_training_data.py
import pandas as pd
import numpy as np
import os
import sys
from itertools import product

# Add project root to path for correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reactor_model.simulation import run_simulation

def generate_training_data(num_samples=500):
    """
    Generates a dataset of deviations and their optimal corrections.
    This will be the data used to train our ML recommender model.
    """
    print("--- Starting Training Data Generation ---")
    
    # 1. Load the Golden Batch as our reference target
    try:
        golden_df = pd.read_csv('data/processed/golden_batch.csv')
    except FileNotFoundError:
        print("Error: golden_batch.csv not found. Please run generate_datasets.py first.")
        return

    training_samples = []
    initial_conditions = [1.0, 3.0, 0.0]

    for i in range(num_samples):
        # --- 2. Simulate a "Problem" Run with Deviations ---
        # Start with golden conditions and add random deviations
        base_T = golden_df['Temperature'].iloc[0]
        base_P = golden_df['Pressure'].iloc[0]
        
        dev_T = base_T + np.random.uniform(-75, 75) # Introduce significant temp deviations
        dev_P = base_P + np.random.uniform(-75, 75) # Introduce significant pressure deviations

        # Pick a random point in time to evaluate the deviation
        eval_time = np.random.randint(20, 80)
        time_points_to_eval = np.arange(0, eval_time + 1)
        
        # Run the simulation with these "bad" conditions
        sim_result_dev = run_simulation(initial_conditions, time_points_to_eval, dev_T, dev_P)
        
        # --- 3. Record the Problem State (our ML model's input) ---
        current_NH3 = sim_result_dev[-1, 2]
        target_NH3 = np.interp(eval_time, golden_df['Time'], golden_df['NH3'])
        deviation = current_NH3 - target_NH3
        
        problem_state = {
            'Time': eval_time,
            'Current_NH3': current_NH3,
            'Deviation': deviation,
            'Current_Temp': dev_T,
            'Current_Press': dev_P
        }

        # --- 4. Search for the Best "Solution" ---
        best_correction = None
        min_future_error = float('inf')
        
        # Define a search space for possible adjustments
        temp_adjustments = [-50, -25, 0, 25, 50]
        press_adjustments = [-50, -25, 0, 25, 50]
        future_time_horizon = np.arange(eval_time, eval_time + 15) # Look 15 seconds ahead

        # Test every possible combination of adjustments
        for T_adj, P_adj in product(temp_adjustments, press_adjustments):
            corrected_T = dev_T + T_adj
            corrected_P = dev_P + P_adj
            
            # Simulate the future with this correction applied
            current_concentrations = sim_result_dev[-1] # Start from where the deviation left off
            future_sim = run_simulation(current_concentrations, future_time_horizon, corrected_T, corrected_P)
            
            # Calculate how far this future path is from the ideal golden path
            future_target_NH3 = np.interp(future_time_horizon, golden_df['Time'], golden_df['NH3'])
            future_error = np.mean((future_sim[:, 2] - future_target_NH3)**2) # Mean Squared Error
            
            if future_error < min_future_error:
                min_future_error = future_error
                best_correction = {
                    'Temp_Adjustment': T_adj,
                    'Press_Adjustment': P_adj
                }
        
        # --- 5. Store the (Problem, Solution) Pair ---
        if best_correction:
            training_samples.append({**problem_state, **best_correction})

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{num_samples} samples...")

    # --- 6. Save the final dataset ---
    output_path = 'data/processed/training_data.csv'
    training_df = pd.DataFrame(training_samples)
    training_df.to_csv(output_path, index=False)
    print(f"\n--- Training data generation complete. ---")
    print(f"Saved {len(training_df)} samples to {output_path}")

if __name__ == "__main__":
    generate_training_data(num_samples=500)
