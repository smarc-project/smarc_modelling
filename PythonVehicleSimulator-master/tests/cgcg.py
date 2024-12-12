import numpy as np
from python_vehicle_simulator.lib import *
from python_vehicle_simulator.vehicles import *

import matplotlib.pyplot

from python_vehicle_simulator.vehicles.SAM import SAM

# Initialize the SAM object
sam = SAM(
    controlSystem="stepInput",
    r_z=10.0,  # Desired depth
    r_psi=30.0,  # Desired heading angle in degrees
    r_rpm=500,  # Desired RPM
    V_current=1.0,  # Current speed
    beta_current=45.0,  # Current direction in degrees
)

# Define the test inputs for ksi, ksi_dot, and ksi_ddot
# Each entry corresponds to the time-varying parameters described in the method documentation
ksi = [
    0.1,   # x_vbs (m)
    0.2,   # x_lcg (m)
    np.deg2rad(5),  # delta_e (stern plane angle, rad)
    np.deg2rad(10), # delta_r (rudder angle, rad)
    0.0, 0.0        # theta_rpm_i (rotation angles of propellers, rad)
]

ksi_dot = [
    0.01,  # x_dot_vbs (m/s)
    0.02,  # x_dot_lcg (m/s)
    np.deg2rad(0.5),  # delta_e_dot (rad/s)
    np.deg2rad(1.0),  # delta_r_dot (rad/s)
    0.0, 0.0          # theta_dot_rpm_i (rad/s)
]

ksi_ddot = [
    0.001,  # x_ddot_vbs (m/s^2)
    0.002,  # x_ddot_lcg (m/s^2)
    np.deg2rad(0.05),  # delta_e_ddot (rad/s^2)
    np.deg2rad(0.1),   # delta_r_ddot (rad/s^2)
    0.0, 0.0           # theta_ddot_rpm_i (rad/s^2)
]

# Run the calculation
results = sam.calculate_center_of_gravity_and_dynamics(ksi, ksi_dot, ksi_ddot)

# Print the results for verification
print("Center of Gravity (CG):", results["r_BG"])
print("Velocity of CG:", results["r_dot_BG"])
print("Acceleration of CG:", results["r_ddot_BG"])
print("Total Moment of Inertia:", results["J_total"])
print("Time Derivative of Moment of Inertia:", results["J_dot_total"])
print("Total Additional Angular Momentum:", results["h_add_total"])
print("Time Derivative of Additional Angular Momentum:", results["h_dot_add_total"])

# # Assertions for basic consistency checks (optional)
# assert isinstance(results["r_BG"], np.ndarray), "r_BG should be a numpy array"
# assert results["r_BG"].shape == (3,), "r_BG should be a 3-element vector"
# assert isinstance(results["J_total"], np.ndarray), "J_total should be a numpy array"
# assert results["J_total"].shape == (3, 3), "J_total should be a 3x3 matrix"