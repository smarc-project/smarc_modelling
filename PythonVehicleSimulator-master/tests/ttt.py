import numpy as np
import sys
from python_vehicle_simulator.lib import *
from python_vehicle_simulator.vehicles import *

import numpy as np
import matplotlib.pyplot
matplotlib.use("TkAgg")

# Define global time array
time = np.linspace(0, 20, 1000)

# Variable pieces
variable_pieces = [
    [  # Variable 1: Mixed signal types with continuity and out-of-bounds handling
        {"interval": (0, 3), "name": "sin", "params": {"freq": 1, "amp": 2}, "time_mode": "absolute"},
        {"interval": (3, 7), "name": "square", "params": {"period": 1, "upper": 3, "lower": -3, "duty": 70}, "continuity": True},
        {"interval": (7, 10), "name": "ramp", "params": {"start_val": 3, "end_val": 1}, "out_of_range": "continue"},
    ],
    [  # Variable 2: Data interpolation with different methods
        {"interval": (0, 5), "name": "data", "params": {"data_points": [(0, 1), (2, 3), (5, 0)], "interp_method": "pchip"}},
        {"interval": (5, 10), "name": "data", "params": {"data_points": [(5, 0), (7, 4), (10, 2)], "interp_method": "spline"}, "out_of_range": "continue"},
    ],
    [  # Variable 3: Custom function with constant out-of-bounds
        {"interval": (0, 5), "name": "custom", "params": {"formula": "t**2 - 3*t + 2"}},
        {"interval": (5, 8), "name": "constant", "params": {"val": 3}, "out_of_range": "zero", "continuity": True},
    ],
]

# Create the signal generator
multi_signal = MultiVariablePiecewiseSignal(time, variable_pieces)

# Query specific time points
query_times = [1, 3.5, 5, 7, 8, 15, 20]  # Time points to query
query_results = multi_signal(query_times)

# Print the queried values
print("Queried Values:")
for i, result in enumerate(query_results):
    print(f"Variable {i + 1}: {result}")

# Evaluate and plot the precomputed signals using multi_signal.time
plt.figure(figsize=(14, 8))
for idx, sig in enumerate(multi_signal.signals):
    plt.plot(multi_signal.time, sig, label=f"Variable {idx + 1}")

plt.xlabel("Time (s)")
plt.ylabel("Signal")
plt.title("Highly Generic Example: Mixed Signal Types with Advanced Features")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

order = [3]  # Arbitrary rotation sequence (Z, X, Y, Z)
angles = [0.7854]  # Corresponding angles in radians


test = np.array([np.zeros(3) , np.ones(3)])

print("test = :\n", test)
# Calculate DCM
dcm = calculate_dcm(order, angles)
print("Direction Cosine Matrix (DCM):\n", dcm)

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([1,2,3])
print(a)
print(b)

A_transpose = b.T
print(A_transpose)


c = Smtrx(b)
print(c)
print(np.matmul(c,[1,0,-1]))




mp = 1.0  # Mass of the payload
m = 1.5  # Mass of the vehicle
r_bp_b = np.array([0.1, 0.2, 0.3])  # Position vector r_bp^b
r_bg_b = np.array([0.05, 0.1, 0.15])  # Position vector r_bg^b
Ib_b = np.eye(3)  # Inertia matrix of the vehicle in the body frame (3x3)

# Call the MRB function
MRB_matrix = MRB_function(mp, m, r_bp_b, r_bg_b, Ib_b)




euler_vector = np.array([1.0, -1.0, -1.0], dtype=np.float64)
euler_vector_norm = euler_vector/np.linalg.norm(euler_vector)
print("Euler [v1, v2, v3]:", euler_vector_norm)

# angle = np.pi / 4  # 45 degrees in radians
angle = np.pi/2

# Convert Euler vector and angle to quaternion
quaternion = euler_to_quaternion(euler_vector, angle)
print("Quaternion [q1, q2, q3, q0]:", quaternion)

# Convert the quaternion back to an Euler vector and angle
euler_vec, angle_recovered = quaternion_to_euler(quaternion)
print("\nRecovered Euler Vector:", euler_vector_norm)
print("Recovered Angle (radians):", angle_recovered*180/math.pi)


dcm = quat_to_dcm_closed_form(quaternion)
print("dcm", dcm)

qq =  dcm_to_quaternion(dcm)
print("qq", qq)


dcm2 = quat_to_dcm_closed_form(qq)
print("dcm", dcm2)



print(MRB_matrix)