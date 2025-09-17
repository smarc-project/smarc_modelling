#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlueROV.py:

   Class for the BlueROV 

   Actuator systems:
    8 thrusters in the heavy configuration to allow 6DoF motions.

   Sensor systems:
   - **IMU**: Inertial Measurement Unit for attitude and acceleration.
   - **DVL**: Doppler Velocity Logger for measuring underwater velocity.
   - **GPS**: For surface position tracking.
   - **Sonar**: For environment sensing during navigation and inspections.

   BlueROV()
       Step input for force and torque control input

Methods:

    [xdot] = dynamics(x, u_ref) returns for integration

    u_ref: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]


References:

    Bhat, S., Panteli, C., Stenius, I., & Dimarogonas, D. V. (2023). Nonlinear model predictive control for hydrobatic AUVs:
        Experiments with the SAM vehicle. Journal of Field Robotics, 40(7), 1840-1859. doi:10.1002/rob.22218.

    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control. 2nd Edition, Wiley.
        URL: www.fossen.biz/wiley

Author:     David Doerner
"""

import numpy as np
import math
from scipy.linalg import block_diag
from smarc_modelling.lib.gnc import *


class SolidStructure:
    """
    Represents the Solid Structure (SS) of the SAM AUV.

    Attributes:
        l_SS: Length of the solid structure (m).
        d_SS: Diameter of the solid structure (m).
        m_SS: Mass of the solid structure (kg).
        p_CSsg_O: Vector from frame C to CG of SS expressed in O (m)
        p_OSsg_O: Vector from CO to CG of SS expressed in O (m)
    """

    def __init__(self, l_ss, d_ss, m_ss, p_CSsg_O, p_OC_O):
        self.l_ss = l_ss
        self.d_ss = d_ss
        self.m_ss = m_ss
        self.p_CSsg_O = p_CSsg_O
        self.p_OSsg_O = p_OC_O + self.p_CSsg_O



# Class Vehicle
class BlueROV():
    """
    SAM()
        Integrates all subsystems of the Small and Affordable Maritime AUV.


    Attributes:
        eta: [x, y, z, q0, q1, q2, q3] - Position and quaternion orientation
        nu: [u, v, w, p, q, r] - Body-fixed linear and angular velocities

    Vectors follow Tedrake's monogram:
    https://manipulation.csail.mit.edu/pick.html#monogram
    """
    def __init__(
            self,
            dt=0.02,
            V_current=0,
            beta_current=0,
    ):
        self.dt = dt # Sim time step, necessary for evaluation of the actuator dynamics

        # Constants
        self.p_OC_O = np.array([0., 0, 0.], float)  # Measurement frame C in CO (O)
        self.D2R = math.pi / 180  # Degrees to radians
        self.rho_w = self.rho = 1026  # Water density (kg/m³)
        self.g = 9.81  # Gravity acceleration (m/s²)

        # Initialize Subsystems:
        self.init_vehicle()

        # Reference values and current
        self.V_c = V_current  # Current water speed
        self.beta_c = beta_current * self.D2R  # Current water direction (rad)

        # Initialize state vectors
        self.nu = np.zeros(6)  # [u, v, w, p, q, r]
        self.eta = np.zeros(7)  # [x, y, z, q0, q1, q2, q3]
        self.eta[3] = 1.0

        # Initialize the AUV model
        self.name = ("BlueROV")

        # Rigid-body mass matrix expressed in CO
        self.m = self.ss.m_ss 
        self.p_OG_O = np.array([0., 0, 0.], float)  # CG w.r.t. to the CO, we
        self.p_OB_O = np.array([0., 0, 0], float)  # CB w.r.t. to the CO

        # Weight and buoyancy
        self.W = self.m * self.g
        self.B = self.W 

        # Inertias from von Benzon 2022
        self.Ix = 0.26
        self.Iy = 0.23
        self.Iz = 0.37


        # Added mass terms
        self.Xdu = 6.36
        self.Ydv = 7.12
        self.Zdw = 18.68
        self.Kdp = 0.189
        self.Mdq = 0.135
        self.Ndr = 0.222

        # Linear Damping coefficients
        self.Xu = 13.7
        self.Yv = 0
        self.Zw = 33.0
        self.Kp = 0
        self.Mq = 0.8
        self.Nr = 0
        
        # Nonlienar Damping coefficients
        self.Xuu = 141.0     # x-damping
        self.Yvv = 217.0 # y-damping
        self.Zww = 190.0# z-damping
        self.Kpp = 1.19 # Roll damping
        self.Mqq = 0.47 # Pitch damping
        self.Nrr = 1.5 # Yaw damping

        # System matrices
        self.MRB = np.diag([self.m, self.m, self.m, self.Ix, self.Iy, self.Iz])
        self.MA = np.diag([self.Xdu, self.Ydv, self.Zdw, self.Kdp, self.Mdq, self.Ndr])
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        self.C = np.zeros((6,6))

        self.D = np.zeros((6,6))
        self.D_lin = np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])
        self.D_nl = np.zeros((6,6))

        self.gamma = 100 # Scaling factor for numerical stability of quaternion differentiation

    def init_vehicle(self):
        """
        Initialize all subsystems based on their respective parameters
        """
        self.ss = SolidStructure(
            l_ss=0.46,
            d_ss=0.58,
            m_ss=13.5,
            p_CSsg_O = np.array([0., 0, 0.]),
            p_OC_O=self.p_OC_O
        )


    def dynamics(self, x, u_ref):
        """
        Main dynamics function for integrating the complete AUV state.

        Args:
            t: Current time
            x: state space vector with [eta, nu, u]
            u_ref: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]

        Returns:
            state_vector_dot: Time derivative of complete state vector
        """
        eta = x[0:7]
        nu = x[7:13]
        u = u_ref

        self.calculate_system_state(nu, eta)
        self.calculate_C()
        self.calculate_D()
        self.calculate_g()
        self.calculate_tau(u)

        np.set_printoptions(precision=3)

        print(f"C: {self.C}")

        nu_dot = self.Minv @ (self.tau - np.matmul(self.C,self.nu_r) - np.matmul(self.D,self.nu_r) - self.g_vec)
        eta_dot = self.eta_dynamics(eta, nu)
        x_dot = np.concatenate([eta_dot, nu_dot])

        return x_dot


    def calculate_system_state(self, x, eta):
        """
        Extract speeds etc. based on state and control inputs
        """
        nu = x

        # Extract Euler angles
        quat = eta[3:7]
        quat = quat/np.linalg.norm(quat)
        self.psi, self.theta, self.phi = quaternion_to_angles(quat) 

        # Relative velocities due to current
        u, v, w, _, _, _ = nu
        u_c = self.V_c * math.cos(self.beta_c - self.psi)
        v_c = self.V_c * math.sin(self.beta_c - self.psi)
        self.nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)
        self.nu_r = nu - self.nu_c

        self.U = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        self.U_r = np.linalg.norm(self.nu_r[:3])

        self.alpha = 0.0
        if abs(self.nu_r[0]) > 1e-6:
            self.alpha = math.atan2(self.nu_r[2], self.nu_r[0])


    def calculate_C(self):
        """
        Calculate Corriolis Matrix
        """
        CRB = m2c(self.MRB, self.nu_r)
        CA = m2c(self.MA, self.nu_r)

        #print(f"nu_r: {self.nu_r}")

        self.C = CRB + CA

    def calculate_D(self):
        """
        Calculate damping
        """
        # Nonlinear damping
        self.D_nl[0,0] = self.Xuu * np.abs(self.nu_r[0])
        self.D_nl[1,1] = self.Yvv * np.abs(self.nu_r[1])
        self.D_nl[2,2] = self.Zww * np.abs(self.nu_r[2])
        self.D_nl[3,3] = self.Kpp * np.abs(self.nu_r[3])
        self.D_nl[4,4] = self.Mqq * np.abs(self.nu_r[4])
        self.D_nl[5,5] = self.Nrr * np.abs(self.nu_r[5])

        self.D = self.D_lin + self.D_nl

    def calculate_g(self):
        """
        Calculate gravity vector
        """
        self.W = self.m * self.g
        self.g_vec = gvect(self.W, self.B, self.theta, self.phi, self.p_OG_O, self.p_OB_O)

    def calculate_tau(self, u):
        """
        All external forces
        Right now, only the control inputs as force and torque around the corresponding axis
        """
        self.tau = u 


    def eta_dynamics(self, eta, nu):
        """
        Computes the time derivative of position and quaternion orientation.

        Args:
            eta: [x, y, z, q0, q1, q2, q3] - Position and quaternion
            nu: [u, v, w, p, q, r] - Body-fixed velocities

        Returns:
            eta_dot: [ẋ, ẏ, ż, q̇0, q̇1, q̇2, q̇3]
        """
        # Extract position and quaternion
        q = eta[3:7]  # [q0, q1, q2, q3] where q0 is scalar part
        q = q/np.linalg.norm(q)

        # Convert quaternion to DCM for position kinematics
        C = quaternion_to_dcm(q)

        # Position dynamics: ṗ = C * v
        pos_dot = C @ nu[0:3]

        ## From Fossen 2021, eq. 2.78:
        om = nu[3:6]  # Angular velocity
        q0, q1, q2, q3 = q
        T_q_n_b = 0.5 * np.array([
                                 [-q1, -q2, -q3],
                                 [q0, -q3, q2],
                                 [q3, q0, -q1],
                                 [-q2, q1, q0]
                                 ])
        q_dot = T_q_n_b @ om + self.gamma/2 * (1 - q.T.dot(q)) * q

        return np.concatenate([pos_dot, q_dot])

