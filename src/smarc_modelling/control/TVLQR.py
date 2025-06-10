import scipy.linalg 
import numpy as np
import casadi as ca

# Add the LQR below -UPDATED TO ONLY INPUT FULL QUATERNIONS BUT ONLY USE THE 3 VECTORIAL
class LQR:
    def __init__(self, dynamics, Ts):
        self.Ts = Ts
        self.dynamics = dynamics
        self.x_lin_prev = np.zeros((12,))
        # State weight matrix
        Q_diag = np.ones(12)
        Q_diag[ 0:3 ] = 1
        Q_diag[ 3:6 ] = 5
        Q_diag[ 6:9] = 1
        Q_diag[9:] = 5
        self.Q = np.diag(Q_diag)


        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        R_diag = np.ones(6)
        R_diag[ :2] = 1e-4
        R_diag[2:4] = 1/50
        R_diag[4: ] = 1e-6
        self.R = np.diag(R_diag)*10

    def create_linearized_dynamics(self, nx: int, nu: int):
        """
        Method to obtain the system jacobians.

        Parameters:
        nx: rows of x vector
        nu: rows of u vector
        
        Returns:
        A (ca.function): State Jacobian
        B (ca.Function): Control Jacobian
        """
        x = ca.MX.sym('x', nx, 1)
        u = ca.MX.sym('u', nu, 1)
        #Discretize the dynamics function
        f_disc = ca.Function('f_disc', [x, u], [x + 0.1 * self.dynamics(x, u)])  # Discretized dynamics function
        A_sym = ca.jacobian(f_disc(x, u), x)
        B_sym = ca.jacobian(f_disc(x, u), u)
        self.A = ca.Function('A_func', [x, u], [A_sym])
        self.B = ca.Function('B_func', [x, u], [B_sym])

        
    def get_numerical_matrices(self, x_lin, u_lin):
        """
        Method to obtain the continuous-time numerated linearized matrices A and B.

        Parameters:
        x_lin (np.array): State to linearize around
        u_lin (np.array): Control to linearize around
        
        Returns:
        Ac (ca.function): Continuous-time state matrix
        Bc (ca.Function): Continuous-time control matrix
        """
        Ac = self.A(x_lin, u_lin)
        Bc = self.B(x_lin, u_lin)
        return Ac, Bc



    def compute_lqr_gain(self, T):
        P = self.Q
        print("Computing LQR gain for T =", T)
        K_list = [None] * (T)

        for t in reversed(range(T)):
            A = self.A_list[t]
            B = self.B_list[t]
            M         = P - P @ B @ scipy.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P
            K_list[t] = scipy.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A  # K = (R + BᵀPB)⁻¹ BᵀPA
                
            P = self.Q + A.T @ M @ A
        return K_list

        
        # P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, Q, R)
        # L = scipy.linalg.inv(R + self.Bd.T @ P @ self.Bd) @ self.Bd.T @ P @ self.Ad


    def solve_k_list(self, trajectory):
        # Since the linearization points are along a trajectory, the reference points is chosen to be the same
        x_lin = trajectory[:, :13]
        u_lin = trajectory[:, 13:]

        x_lin = np.delete(x_lin, 3, axis=1)  # Remove the scalar quaternion
        
        self.A_list, self.B_list = [], []
        for i in range(trajectory.shape[0]):
            Ac, Bc = self.get_numerical_matrices(x_lin[i,:], u_lin[i,:])      # Create matrix A and B in continuous time
            self.A_list.append(Ac)
            self.B_list.append(Bc)
        self.K_list = self.compute_lqr_gain(trajectory.shape[0])            # Calculate the feedback gain

    
    def get_u(self, x, trajectory, i):
        # Calculate control input
        # Since delta_u =-L*delta_x, delta_u = u-u_ref --> u = -L*delta_x + u_ref
        u = -self.K_list[i] @ (self.x_error(x, trajectory[i, :13])) + trajectory[i, 13:].reshape(-1, 1)


        # Convert output from casadi.DM to np.array 
        u = np.array(u).flatten()

        return u
    
    def x_error(self, x, ref):
        """
        Calculates the state error.
        
        :param x: State vector
        :param ref: Reference vector
        :return: error vector
        """
        # Extract the reference quaternion
        q_ref = ref[3:7]

        # Extract current quaternion
        q = x[3:7]

        # Since unit quaternion, quaternion inverse is equal to its conjugate
        q_conj = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
        q_ref = q_conj
        # q_error = q_ref @ q^-1
        q_w = q_ref[0] * q[0] - q_ref[1] * q[1] - q_ref[2] * q[2] - q_ref[3] * q[3]
        q_x = q_ref[0] * q[1] + q_ref[1] * q[0] + q_ref[2] * q[3] - q_ref[3] * q[2]
        q_y = q_ref[0] * q[2] - q_ref[1] * q[3] + q_ref[2] * q[0] + q_ref[3] * q[1]
        q_z = q_ref[0] * q[3] + q_ref[1] * q[2] - q_ref[2] * q[1] + q_ref[3] * q[0]


        q_error = ca.vertcat(q_w, q_x, q_y, q_z)

        q_error = q_error / ca.norm_2(q_error)

        q_error = ca.vertcat(q_error[1], q_error[2], q_error[3])

        pos_error = x[:3] - ref[:3] 
        vel_error = x[7:13] - ref[7:13]
        x_error = ca.vertcat(pos_error, q_error, vel_error)

        return x_error
    

    def x_error2(self, x, ref):
        """
        Calculates the state deviation.
        
        :param x: State vector
        :param ref: Reference vector
        :return: error vector
        """
        q1 = ref[3:7]
        q2 = x[3:7]
        # Sice unit quaternion, quaternion inverse is equal to its conjugate
        q_conj = ca.vertcat(q2[0], -q2[1], -q2[2], -q2[3])
        q2 = q_conj
        
        # q_error = q1 @ q2^-1
        q_w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q_x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q_y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q_z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        q_error = ca.vertcat(q_x, q_y, q_z)

        pos_error = x[:3] - ref[:3]
        vel_error = x[7:13] - ref[7:13]
        

        # If the error for terminal cost is calculated, don't include delta_u
        x_error = ca.vertcat(pos_error, q_error, vel_error)
        return x_error
