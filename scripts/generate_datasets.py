# scripts/generate_datasets.py
import pandas as pd
import numpy as np
import os
import sys

# Add the project's root directory to the Python path
# This allows us to import from the 'reactor_model' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reactor_model.simulation import run_simulation

def generate_dataset(file_path, T, P, noise_level=0.01):
    """
    Generates a single reactor dataset for a given Temperature and Pressure,
    adds realistic noise, and saves it as a CSV file.
    """
    # Define standard initial conditions and time
    # Concentrations: [N2, H2, NH3]
    initial_conditions = [1.0, 3.0, 0.0]
    time_points = np.linspace(0, 100, 101) # 101 points for 1-second intervals

    # Run the perfect simulation to get the "true" data
    true_data = run_simulation(initial_conditions, time_points, T, P)

    # Simulate a noisy sensor by adding random noise to our measurement (NH3)
    noisy_NH3 = true_data[:, 2] + np.random.normal(0, noise_level, len(time_points))
    noisy_NH3[noisy_NH3 < 0] = 0 # Concentration can't be negative

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'Time': time_points,
        'N2': true_data[:, 0],      # True simulated concentration of N2
        'H2': true_data[:, 1],      # True simulated concentration of H2
        'NH3': noisy_NH3,           # The noisy NH3 reading from our "sensor"
        'Temperature': T,
        'Pressure': P
    })

    # Create the directory if it doesn't exist and save the file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Successfully generated dataset: {file_path}")

if __name__ == "__main__":
    # --- Define Conditions for Different Simulation Runs ---

    # 1. The "Golden Batch" - Our target optimal conditions.
    golden_temp = 700  # Kelvin (approx. 427°C)
    golden_pressure = 200 # Atmospheres
    generate_dataset(
        'data/processed/golden_batch.csv',
        T=golden_temp,
        P=golden_pressure,
        noise_level=0.005 # Less noise for our ideal path reference
    )

    # 2. A deviating run with temperature that is too high
    generate_dataset(
        'data/raw/run_high_temp.csv',
        T=775, # Higher temp hurts the equilibrium yield
        P=golden_pressure # <-- This was the line with the typo
    )

    # 3. A deviating run with pressure that is too low
    generate_dataset(
        'data/raw/run_low_pressure.csv',
        T=golden_temp,
        P=125 # Lower pressure slows the reaction
    )

    print("\n--- All datasets generated. ---")


