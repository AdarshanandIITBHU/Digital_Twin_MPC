# reactor_model/simulation.py
import numpy as np
from scipy.integrate import odeint

def ammonia_synthesis_kinetics(C, t, T, P):
    """
    Defines the Ordinary Differential Equations (ODEs) for the Haber-Bosch process.
    N2 + 3H2 <=> 2NH3
    This function describes the rate of change of concentrations over time.
    """
    # Unpack the concentration array [C_N2, C_H2, C_NH3]
    C_N2, C_H2, C_NH3 = C
    
    # --- Model Parameters (simplified for this simulation) ---
    A = 1.0e5  # Pre-exponential factor (controls overall rate)
    Ea = 5.0e4 # Activation energy in Joules/mol (temperature sensitivity)
    R = 8.314  # Ideal gas constant (J/mol*K)
    
    # --- Rate Constant Calculation ---
    # Arrhenius equation to model how temperature affects the reaction rate
    k1 = A * np.exp(-Ea / (R * T))
    
    # --- Equilibrium Constant Calculation ---
    # This models the balance between forward and reverse reactions
    K_eq = 1.0e-5 * np.exp(4000/T) * P**-0.5

    # --- Rate Laws ---
    # These equations define the speed of the forward and reverse reactions
    forward_rate = k1 * C_N2 * (C_H2**1.5)
    reverse_rate = (k1 / K_eq) * (C_NH3**2) / (C_H2**1.5)
    
    # The net rate of reaction
    net_reaction_rate = forward_rate - reverse_rate
    
    # --- ODEs based on stoichiometry ---
    # How each chemical's concentration changes based on the net rate
    dC_N2_dt = -net_reaction_rate
    dC_H2_dt = -3 * net_reaction_rate
    dC_NH3_dt = 2 * net_reaction_rate
    
    return [dC_N2_dt, dC_H2_dt, dC_NH3_dt]

def run_simulation(initial_conditions, time_points, T, P):
    """
    Wrapper function to solve the ODEs for the given conditions.
    """
    solution = odeint(ammonia_synthesis_kinetics, initial_conditions, time_points, args=(T, P))
    return solution

