# reactor_model/kalman_filter.py
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from .simulation import ammonia_synthesis_kinetics
from scipy.integrate import odeint

def f_x(x, dt, T, P):
    """
    State Transition Function (fx): Predicts the next state.
    It uses our ODE model to predict how concentrations change over one time step 'dt'.
    """
    return odeint(ammonia_synthesis_kinetics, x, [0, dt], args=(T, P))[1]

def h_x(x):
    """
    Measurement Function (hx): Connects the state to the sensor reading.
    Our state is [N2, H2, NH3], but our sensor only measures NH3.
    """
    return [x[2]]

def run_ukf_tracker(measurements, initial_conditions, T, P, noise_levels):
    """
    Initializes and runs the Unscented Kalman Filter.
    It takes noisy measurements and produces a clean estimate of the true state.
    """
    # UKF setup for a 3-variable state and 1-variable measurement
    points = MerweScaledSigmaPoints(n=3, alpha=.1, beta=2., kappa=-1)
    ukf = UKF(dim_x=3, dim_z=1, dt=1.0, hx=h_x, fx=f_x, points=points)

    ukf.x = np.array(initial_conditions) # Set initial state guess

    # --- Set Covariance Matrices (Tuning the Filter) ---
    # P: State Covariance - Our uncertainty about the initial state.
    ukf.P = np.diag([.01, .01, .01])
    # R: Measurement Noise - How much we trust the sensor. (Higher value = less trust)
    ukf.R = np.diag([noise_levels['measurement']**2])
    # Q: Process Noise - How much we trust our model. (Higher value = less trust)
    ukf.Q = np.diag([noise_levels['process']] * 3)

    # --- Run the Predict-Update Loop ---
    ukf_estimates = []
    for z in measurements:
        ukf.predict(T=T, P=P) # Predict the next state using the model
        ukf.update(z)         # Correct the prediction with the real sensor measurement
        ukf_estimates.append(ukf.x.copy())

    return np.array(ukf_estimates)

