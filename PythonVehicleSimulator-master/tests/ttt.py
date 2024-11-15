import numpy as np
import sys
from python_vehicle_simulator.lib import *
from python_vehicle_simulator.vehicles import *

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