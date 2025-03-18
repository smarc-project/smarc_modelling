import scipy.linalg 
import numpy as np
import casadi as ca


# Add the LQR below
class LQR:
    def __init__(self, model):
        Q = np.eye(19)
        R = np.eye(6)
        self.model = model


    def create_linearized_dynamics(self, x_lin, u_lin):
        """
        Function to create continuous-time linearized dynamics.
        """
        x_sym = ca.MX.sym('x', 19, 1)
        u_sym = ca.MX.sym('u', 6, 1)
        
        # Create Casadi functions to calculate jacobian
        self.dfdx = ca.Function('dfdx', [x_sym, u_sym], [ca.jacobian(self.model.dynamics(x_sym, u_sym), x_sym)])
        self.dfdu = ca.Function('dfdu', [x_sym, u_sym], [ca.jacobian(self.model.dynamics(x_sym, u_sym), u_sym)])
        # A_d_sym, Bd_sym = self.continuous_to_discrete(self.Ac_sym, self.Bc_sym, dt = 0.01)

        # f(x_lin, u_lin) is the same as self.model.dynamics(x_lin, u_lin)
        A = ca.horzcat(self.dfdx(x_lin, u_lin), self.model.dynamics(x_lin, u_lin) - self.dfdx(x_lin, u_lin) @ x_lin)
        
        # To construct the A and B matrices following the procedure, a zeros and a ones matrices needs to be declared
        A_zero = ca.MX.zeros(self.dfdx.size1_out(), self.dfdx.size2_out())
        A_ones = ca.MX.ones(self.model.dynamics.size1_out(), self.model.dynamics.size2_out())
        B_zero = ca.MX.zeros(self.dfdu.size1_out(), self.dfdu.size2_out())

        bottom = ca.horzcat(A_zero, A_ones)
        self.Ac = ca.vertcat(A, bottom)
        self.Bc = ca.vertcat(self.dfdu(x_lin, u_lin), B_zero)

        return self.Ac, self.Bc
        
    def continuous_to_discrete(self, A, B, dt):
        """
        Convert continuous-time system matrices (A, B) to discrete-time (A_d, B_d) using zero-order hold.
        
        Parameters:
        A (ca.MX): Continuous-time state matrix
        B (ca.MX): Continuous-time input matrix
        dt (float): Sampling time
        
        Returns:
        A_d (ca.MX): Discrete-time state matrix
        B_d (ca.MX): Discrete-time input matrix
        """
        # Augmented matrix
        n = A.size1()
        m = B.size2()
        M = ca.vertcat(ca.horzcat(A, B), ca.horzcat(ca.MX.zeros(m, n), ca.MX.eye(m)))
        
        # Matrix exponential
        M_exp = ca.expm(M * dt)
        
        # Extract discrete-time matrices
        A_d = M_exp[:n, :n]
        A_d2 = ca.expm(A * dt)
        print(A_d, A_d2)
        B_d = M_exp[:n, n:]
        
        return A_d, B_d

    def compute_lqr_gain(self, A, B, Q, R):
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.L = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    def solve(self, x):
        dt = 0.1
        Q = np.eye(19)
        R = np.eye(6)
        A, B = self.create_linearized_dynamics(x, u)
        self.compute_lqr_gain(A, B, Q, R)
        u = self.L @ x

        x_dot = A @ x + B @ u
        x += x_dot * dt  # Update the state (Euler method)

        return x
    
    

# u = -k*x
# x_dot = auv_dynamics(x, u)
# x += x_dot * dt  # Update the state (Euler method)