import scipy.linalg 
import numpy as np
import casadi as ca


# Add the LQR below
class LQR:
    def __init__(self):
        Q = np.eye(19)
        R = np.eye(6)

    def linearize_system(self, model):
        A = 11
        B = 22
        pass

    def compute_lqr_gain(self, A, B, Q, R):
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.L = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    def solve(self, x):
        u = -self.L @ x

        x_dot = linearized_discrete_dynamics(x, u)
        x += x_dot * dt  # Update the state (Euler method)
        return x

# u = -k*x
# x_dot = auv_dynamics(x, u)
# x += x_dot * dt  # Update the state (Euler method)