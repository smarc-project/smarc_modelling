import scipy.linalg 
import numpy as np
import casadi as ca


# Add the LQR below
class LQR:
    def __init__(self, model, Ts):
        self.P = 0
        self.Ts = Ts
        self.model = model


    def create_linearized_dynamics(self, x_lin, u_lin):
        """
        Function to create continuous-time linearized dynamics.
        """
        nx = np.size(x_lin, 0)
        nu = np.size(u_lin, 0)    
        x_sym = ca.MX.sym('x', nx, 1)
        u_sym = ca.MX.sym('u', nu, 1)
        
        # Create Casadi functions to calculate jacobian
        # self.dfdx = ca.Function('dfdx', [x_sym, u_sym], [ca.jacobian(self.model(x_sym, u_sym), x_sym)])
        # self.dfdu = ca.Function('dfdu', [x_sym, u_sym], [ca.jacobian(self.model(x_sym, u_sym), u_sym)])
        # print(self.dfdx)
        # A_d_sym, Bd_sym = self.continuous_to_discrete(self.Ac_sym, self.Bc_sym, dt = 0.01)


        # f(x_lin, u_lin) is the same as self.model.dynamics(x_lin, u_lin)
        
        # To construct the A and B matrices following the procedure, a zeros and a ones matrices needs to be declared
        A_zero = ca.MX.zeros(1 ,nx)
        A_ones = ca.MX.ones(1)
        B_zero = ca.MX.zeros(1, nu)

        self.Ac = ca.Function('Ac', [x_sym, u_sym], [ca.vertcat(ca.horzcat(ca.jacobian(self.model(x_sym, u_sym), x_sym), 
                                                                           self.model(x_sym, u_sym) - ca.jacobian(self.model(x_sym, u_sym), x_sym) @ x_sym),
                                                                ca.horzcat(A_zero, A_ones))
                                                                ])
        self.Bc = ca.Function('Bc', [x_sym, u_sym], [ca.vertcat(ca.jacobian(self.model(x_sym, u_sym), u_sym), B_zero)])

        print(self.Ac(np.ones(nx), np.ones(nu)).size())
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

    def compute_lqr_gain(self, A, B):
        Q = np.eye(20)
        R = np.eye(6)*1e-6

        condition_number = np.linalg.cond(A)
        print("Condition number:", condition_number)
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.L = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return self.L

    def solve(self, x):
        Q = np.eye(19)
        R = np.eye(6)
        A, B = self.create_linearized_dynamics(x, u)
        self.compute_lqr_gain(A, B, Q, R)
        u = -self.L @ x

        x_dot = A @ x + B @ u
        x += x_dot * self.Ts  # Update the state (Euler method)

        return u
    
    

# u = -k*x
# x_dot = auv_dynamics(x, u)
# x += x_dot * dt  # Update the state (Euler method)