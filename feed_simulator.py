# feed_simulator.py
import pandas as pd
import time
import os
import argparse

# This is the file the dashboard will constantly read for new data
LIVE_DATA_FILE = "live_data.csv"

def simulate_feed(source_file):
    """
    Reads a chosen dataset (e.g., 'run_high_temp.csv') and writes it
    out to 'live_data.csv' line-by-line to simulate a real-time data stream.
    """
    print(f"--- Starting Live Feed Simulator from source: '{source_file}' ---")

    try:
        source_data = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"Error: Source file not found at '{source_file}'")
        print("Please run 'python scripts/generate_datasets.py' first.")
        return

    # Create/overwrite the live file with just the header
    with open(LIVE_DATA_FILE, "w") as f:
        f.write(",".join(source_data.columns) + "\n")

    print(f"Created '{LIVE_DATA_FILE}'. The dashboard will now watch this file.")
    print("Starting data feed... (Press Ctrl+C to stop)")

    # Loop through the source data and append one row every second
    for index, row in source_data.iterrows():
        with open(LIVE_DATA_FILE, "a") as f:
            f.write(",".join(map(str, row.values)) + "\n")
        print(f"Sent data for Time = {row['Time']:.0f}s")
        time.sleep(1)

    print("\n--- Feed complete. ---")

if __name__ == "__main__":
    # Set up command-line arguments to choose which dataset to stream
    parser = argparse.ArgumentParser(description="Simulate a live reactor data feed.")
    parser.add_argument(
        '--source',
        type=str,
        default='data/raw/run_high_temp.csv',
        help="Path to the source CSV file for the simulation."
    )
    args = parser.parse_args()
    simulate_feed(args.source)