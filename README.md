Ammonia Synthesis Digital Twin with ML-Based Process Control
This project implements a full-fledged digital twin for a simulated chemical batch reactor executing the Haber-Bosch process for ammonia synthesis (N₂ + 3H₂ ⇌ 2NH₃).

The system provides real-time monitoring of a "live" simulated reactor, compares its performance against a pre-defined optimal path (a "Golden Batch"), and uses a machine learning model to provide prescriptive recommendations for corrective actions to steer the process back towards optimal conditions.

🚀 Features
Physics-Based Simulation: A core reactor model built on Ordinary Differential Equations (ODEs) that accurately simulates the kinetics of the Haber-Bosch process.

Real-time State Estimation: An Unscented Kalman Filter (UKF) processes noisy sensor data to provide a clean, stable estimate of the concentrations of all chemicals (N₂, H₂, and NH₃).

Live Monitoring Dashboard: An interactive web application built with Streamlit that visualizes the live reactor trajectory against the "Golden Batch" path.

Deviation Detection & Alerting: The system automatically detects when the live process deviates significantly from the optimal path and flags a warning.

ML-Powered Recommendations: An XGBoost model, trained on thousands of simulated scenarios, provides real-time, prescriptive recommendations for control actions (Temperature and Pressure adjustments) to correct deviations.

Modular Data Pipeline: Separate, easy-to-run scripts for generating optimal paths, simulated faulty runs, and the specialized training data required for the machine learning model.

🛠️ System Architecture & Workflow
The project operates through a seamless flow of data and logic between its components:

Data Generation (Offline Phase):

scripts/generate_datasets.py: Runs the core ODE simulator to create the ideal "Golden Batch" trajectory and several deviating datasets (e.g., high temperature, low pressure) that simulate faulty reactor runs.

scripts/generate_training_data.py: Creates a rich dataset of "problem-solution" pairs. It simulates thousands of deviations and, for each one, determines the optimal corrective action (e.g., "decrease temp by 50K") required to get back on track.

Model Training (Offline Phase):

scripts/train_model.py: Loads the specialized training data and trains two XGBoost regressor models—one to predict temperature adjustments and one for pressure adjustments. The trained models are saved as .joblib files.

Live Simulation & Monitoring (Online Phase):

feed_simulator.py: Mimics a live data stream from the reactor by reading a faulty run CSV and writing its rows to live_data.csv one by one, with a one-second delay.

dashboard/app.py: The main application orchestrates the live monitoring:

It reads the live_data.csv file every second.

It uses the Unscented Kalman Filter (kalman_filter.py) to get a clean estimate of the reactor's true state.

It plots the estimated state against the "Golden Batch" path.

If a sustained deviation is detected, it feeds the current "problem state" to the pre-trained ML models.

It displays the recommended corrective actions from the models directly on the dashboard.

💻 Technology Stack
Core Language: Python 3.9+

Simulation & Data Science: NumPy, Pandas, SciPy

Machine Learning: Scikit-learn, XGBoost, Joblib

State Estimation: FilterPy

Dashboard: Streamlit

Plotting: Matplotlib

⚙️ Setup and Installation
Follow these steps to set up the project locally.

1. Clone the Repository

git clone [https://github.com/AdarshanandIITBHU/Digital_Twin_MPC.git](https://github.com/AdarshanandIITBHU/Digital_Twin_MPC.git)
cd Digital_Twin_MPC

2. Create and Activate a Virtual Environment

Windows:

python -m venv .venv
.\.venv\Scripts\activate

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

▶️ How to Run the Project
To see the digital twin in action, you need to first generate the necessary data and models, then run the live simulation.

Important: The following commands must be run from the root directory of the project (Ammonia_Digital_Twin/).

Step 1: Generate Datasets
This creates the "Golden Batch" and the simulated faulty runs.

python scripts/generate_datasets.py

Step 2: Generate Training Data for the ML Model
This may take a minute or two as it runs thousands of simulations.

python scripts/generate_training_data.py

Step 3: Train the ML Models
This will create and save temp_model.joblib and press_model.joblib.

python scripts/train_model.py

Step 4: Run the Application
This requires two separate terminals.

In your first terminal, start the live data feed simulator:

# This will simulate a high-temperature fault by default
python feed_simulator.py

# Optional: Simulate a low-pressure fault instead
# python feed_simulator.py --source data/raw/run_low_pressure.csv

In your second terminal, launch the Streamlit dashboard:

streamlit run dashboard/app.py

The dashboard will open in your web browser. After about 5 seconds of deviation, you will see the ML recommendations appear on the left-hand panel.

📂 File Structure
├── data/
│   ├── raw/                # Stores simulated faulty reactor run data.
│   └── processed/          # Stores the "Golden Batch" and ML training data.
├── dashboard/
│   └── app.py              # The main Streamlit dashboard application.
├── reactor_model/
│   ├── simulation.py       # The core ODE model for the Haber-Bosch process.
│   ├── kalman_filter.py    # Unscented Kalman Filter for state estimation.
│   ├── temp_model.joblib   # Saved ML model for temperature recommendations.
│   └── press_model.joblib  # Saved ML model for pressure recommendations.
├── scripts/
│   ├── generate_datasets.py
│   ├── generate_training_data.py
│   └── train_model.py
├── feed_simulator.py       # Mimics a live data feed from the reactor.
└── requirements.txt        # Project dependencies.
