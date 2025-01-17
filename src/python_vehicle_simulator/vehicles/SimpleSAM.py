#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM.py:

   Class for the SAM (Small and Affordable Maritime) cylinder-shaped autonomous underwater vehicle (AUV),
   designed for agile hydrobatic maneuvers, including obstacle avoidance, inspections, docking, and under-ice operations.
   The SAM AUV is controlled using counter-rotating propellers, a thrust vectoring system, a variable buoyancy system (VBS),
   and adjustable battery packs for center of gravity (c.g.) control. It is equipped with sensors such as IMU, DVL, GPS, and sonar.

   The length of the AUV is 1.5 m, the cylinder diameter is 19 cm, and the mass of the vehicle is 31.9 kg.
   It has a maximum speed of 2.5 m/s, which is obtained when the propellers run at 1525 rpm in zero currents.
   SAM was developed by the Swedish Maritime Robotics Center and is underactuated, meaning it has fewer control inputs than
   degrees of freedom. The control system uses both static and dynamic actuation for different maneuvers.

   Actuator systems:
   1. **Counter-Rotating Propellers**: Two propellers used for propulsion, rotating in opposite directions to balance the roll and provide forward thrust.
   2. **Thrust Vectoring System**: Propellers can be deflected horizontally (rudder-like) and vertically (stern-plane-like) with angles up to ±7°, enabling agile maneuvers.
   3. **Variable Buoyancy System (VBS)**: Allows for depth control by altering buoyancy through water intake and release.
   4. **Adjustable Center of Gravity (c.g.) Control**: Movable battery packs adjust the longitudinal and transversal c.g. positions, allowing for pitch and roll control.
   5. **Rotating Counterweights**: Provides static roll control by shifting weight in the transverse direction.

   Sensor systems:
   - **IMU**: Inertial Measurement Unit for attitude and acceleration.
   - **DVL**: Doppler Velocity Logger for measuring underwater velocity.
   - **GPS**: For surface position tracking.
   - **Sonar**: For environment sensing during navigation and inspections.

   SAM()
       Step input for tail rudder, stern plane, and propeller revolutions.

   SAM('depthHeadingAutopilot',z_d,psi_d,rpm_1,rpm_2,V_c,beta_c)
        z_d:    desired depth (m), positive downwards
        psi_d:  desired yaw angle (deg)
        rpm_1:  desired propeller revolution for propeller 1 (rpm)
        rpm_2:  desired propeller revolution for propeller 2 (rpm)
        V_c:    current speed (m/s)
        beta_c: current direction (deg)

Methods:

    [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime ) returns
        nu[k+1] and u_actual[k+1] using Variable RK method(Possibility for other Methods). The control input is:

            u_control = [ delta_r   rudder angle (rad)
                         delta_s    stern plane angle (rad)
                         rpm_1      propeller 1 revolution (rpm)
                         rpm_2      propeller 2 revolution (rpm)
                         vbs        variable buoyancy system control
                         lcg        longitudinal center of gravity adjustment ]

        - **delta_r**: Rudder angle for horizontal thrust vectoring, used to control yaw (turning left/right).
        - **delta_s**: Stern plane angle for vertical thrust vectoring, used to control pitch (nose up/down).
        - **rpm_1**: Propeller RPM for the first (counter-rotating) propeller, controlling forward thrust.
        - **rpm_2**: Propeller RPM for the second (counter-rotating) propeller, also controlling forward thrust and balancing roll.
        - **vbs**: Variable buoyancy system control, which adjusts buoyancy to control depth.
        - **lcg**: Longitudinal center of gravity adjustment by moving the battery pack to control pitch.

    u = depthHeadingAutopilot(eta,nu,sampleTime)
        Simultaneous control of depth and heading using PID and Sliding Mode Controllers (SMC).
        The propeller RPMs are given as step commands, while thrust vectoring and c.g. control are used for precision adjustments.

    u = stepInput(t) generates tail rudder, stern planes, and RPM step inputs for both propellers.

References:

    Bhat, S., Panteli, C., Stenius, I., & Dimarogonas, D. V. (2023). Nonlinear model predictive control for hydrobatic AUVs:
        Experiments with the SAM vehicle. Journal of Field Robotics, 40(7), 1840-1859. doi:10.1002/rob.22218.

    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control. 2nd Edition, Wiley.
        URL: www.fossen.biz/wiley


Author:     Omid Mirzaeedodangeh

Refactored: David Doerner
"""
import sys

import numpy as np
import math
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d
from scipy.linalg import block_diag
from python_vehicle_simulator.lib.control import integralSMC
from python_vehicle_simulator.lib.gnc import *


#class SolidStructure:
#    """
#    Represents the Solid Structure (SS) of the SAM AUV.
#
#    Attributes:
#        l_SS: Length of the solid structure (m).
#        d_SS: Diameter of the solid structure (m).
#        m_SS: Mass of the solid structure (kg).
#        J_SS_c: Inertia tensor of the solid structure relative to the body frame (kg·m²).
#        r_SS_c: Position vector of the SS center of gravity (CG) relative to the central frame (m).
#    """
#
#    def __init__(self, l_SS, d_SS, m_SS, J_SS_c, r_SS_c):
#        self.l_SS = l_SS
#        self.d_SS = d_SS
#        self.m_SS = m_SS
#        self.J_SS_c = J_SS_c
#        self.r_SS_c = r_SS_c
#
#
class VariableBuoyancySystem:
    """
    VariableBuoyancySystem Class

    Represents the Variable Buoyancy System (VBS) of the AUV.

    Parameters:
        -d_vbs (float): Diameter of the VBS (m).
        -l_vbs_l (float): Length of the VBS capsule (m).
        -h_vbs (float): Vertical offset of the VBS CG (m).
        -l_vbs_b (float): Horizontal offset length of the VBS CG (m).
        -m_vbs_sh (float): Mass of the VBS shaft (kg).
        -r_vbs_sh_cg (list or np.array): Vector from the shaft center to the end boundary of the water in the VBS (m).
        -J_vbs_sh_cg (np.array): Moment of inertia of the VBS shaft around its CG (3x3 matrix).
    """

    def __init__(self, d_vbs, l_vbs_l, h_vbs, l_vbs_b, m_vbs_sh, r_vbs_sh_cg, J_vbs_sh_cg):
        # Physical parameters
        self.d_vbs = d_vbs  # Diameter of VBS (m)
        self.l_vbs_l = l_vbs_l  # Length of VBS capsule (m)
        self.h_vbs = h_vbs  # Vertical offset of VBS CG (m)
        self.l_vbs_b = l_vbs_b  # Horizontal offset of VBS CG (m)
        self.m_vbs_sh = m_vbs_sh  # Mass of VBS shaft (kg)
        self.r_vbs_sh_cg = r_vbs_sh_cg  # Vector from shaft center to water boundary (m)
        self.J_vbs_sh_cg = J_vbs_sh_cg  # Moment of inertia of VBS shaft (kg·m²)

        # Motion bounds
        self.x_vbs_min = 0  # Minimum VBS position (m)
        self.x_vbs_max = l_vbs_l  # Maximum VBS position (m)
        self.x_vbs_dot_min = -0.1  # Maximum retraction speed (m/s)
        self.x_vbs_dot_max = 0.1  # Maximum extension speed (m/s)

#
#class LongitudinalCenterOfGravityControl:
#    """
#    Represents the Longitudinal Center of Gravity Control (LCG) of the SAM AUV.
#
#    Attributes:
#        l_lcg_l: Length of the LCG structure along the x-axis (m).
#        l_lcg_r: Maximum position of the LCG in the x-direction (m).
#        l_lcg_b: Additional offset length along the x-axis (m).
#        h_lcg: Vertical offset of the CG along the z-axis relative to the central frame (m).
#        m_lcg: Mass of the LCG (kg).
#        h_lcg_dim: Height of the LCG structure (m).
#        d_lcg: Width of the LCG structure (m).
#    """
#
#    def __init__(self, l_lcg_l, l_lcg_r, l_lcg_b, h_lcg, m_lcg, h_lcg_dim, d_lcg):
#        # Physical parameters
#        self.l_lcg_l = l_lcg_l  # Length of LCG structure (m)
#        self.l_lcg_r = l_lcg_r  # Maximum x-direction position (m)
#        self.l_lcg_b = l_lcg_b  # Additional x-axis offset (m)
#        self.h_lcg = h_lcg  # Vertical CG offset (m)
#        self.m_lcg = m_lcg  # Mass of LCG (kg)
#        self.h_lcg_dim = h_lcg_dim  # Height of LCG structure (m)
#        self.d_lcg = d_lcg  # Width of LCG structure (m)
#
#        # Motion bounds
#        self.x_lcg_min = 0  # Minimum LCG position (m)
#        self.x_lcg_max = l_lcg_r  # Maximum LCG position (m)
#        self.x_lcg_dot_min = -0.1  # Maximum retraction speed (m/s)
#        self.x_lcg_dot_max = 0.1  # Maximum extension speed (m/s)
#
#
#class ThrusterShaft:
#    """
#    Represents the Thruster Shaft (Tsh) of the SAM AUV.
#
#    Attributes:
#        l_t_sh: Length of the thruster shaft (m).
#        r_t_sh_t: Position vector of the CG relative to the thruster frame origin (m).
#        m_t_sh: Mass of the thruster shaft (kg).
#        J_t_sh_t: Inertia tensor of the thruster shaft relative to the thruster frame (kg·m²).
#    """
#
#    def __init__(self, l_t_sh, r_t_sh_t, m_t_sh, J_t_sh_t):
#        # Physical parameters
#        self.l_t_sh = l_t_sh  # Length of thruster shaft (m)
#        self.r_t_sh_t = r_t_sh_t  # CG position vector (m)
#        self.m_t_sh = m_t_sh  # Mass of thruster shaft (kg)
#        self.J_t_sh_t = J_t_sh_t  # Inertia tensor (kg·m²)
#
#        # Control surface angle bounds
#        self.delta_s_min = -15 * np.pi / 180  # Min stern plane angle (rad)
#        self.delta_s_max = 15 * np.pi / 180  # Max stern plane angle (rad)
#        self.delta_r_min = -15 * np.pi / 180  # Min rudder angle (rad)
#        self.delta_r_max = 15 * np.pi / 180  # Max rudder angle (rad)
#
#        # Control surface angular rate bounds
#        self.delta_s_dot_min = -10 * np.pi / 180  # Max stern plane rate (rad/s)
#        self.delta_s_dot_max = 10 * np.pi / 180
#        self.delta_r_dot_min = -10 * np.pi / 180  # Max rudder rate (rad/s)
#        self.delta_r_dot_max = 10 * np.pi / 180
#
#
class Propellers:
    """
    Represents the Propellers (TP) of the SAM AUV.

    Attributes:
        n_p: Number of propellers.
        l_t_p (np.array): List of fixed offsets of each propeller along the x-axis relative to the thruster frame (m).
        m_t_p (np.array): List of masses of each propeller (kg).
        r_t_p: List of CG position vectors of each propeller (np.array) relative to the propeller frame (m).
        r_t_p_sh: List of each propeller location on thruster shaft (np.array) relative to the thruster frame (m).
        J_t_p: List of inertia tensors of each propellers (np.array) in the propeller frame (kg·m²).
    """

    def __init__(self, n_p, l_t_p, m_t_p, r_t_p, r_t_p_sh, J_t_p):
        # Physical parameters
        self.n_p = n_p  # Number of propellers
        self.l_t_p = l_t_p  # Fixed x-axis offsets list
        self.m_t_p = m_t_p  # Mass list
        self.r_t_p = r_t_p  # CG position vectors list
        self.r_t_p_sh = r_t_p_sh  # Shaft center locations list
        self.J_t_p = J_t_p  # Inertia tensors list

        # RPM bounds
        self.rpm_min = np.zeros(n_p) - 1525  # Min RPM per propeller
        self.rpm_max = np.zeros(n_p) + 1525  # Max RPM per propeller
        self.rpm_dot_min = np.zeros(n_p) - 100  # Max deceleration (RPM/s)
        self.rpm_dot_max = np.zeros(n_p) + 100  # Max acceleration (RPM/s)



# Class Vehicle
class SimpleSAM:
    """
    SAM()
        Integrates all subsystems of the Small and Affordable Maritime AUV.

    Control Modes:
        'depthHeadingAutopilot': Depth and heading autopilots
        'stepInput': Step inputs for VBS, LCG, stern planes, rudder, and propellers

    Attributes:
        eta: [x, y, z, q1, q2, q3, q0] - Position and quaternion orientation
        nu: [u, v, w, p, q, r] - Body-fixed linear and angular velocities
        ksi: Time-varying parameters [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        ksi_dot: Time derivatives of ksi
    """

    def __init__(
            self,
            V_current=0,
            beta_current=0,
    ):
        # Initialize Subsystems:
        self.init_vehicle()

        # Constants
        self.r_cb = np.array([0.75, 0, -0.06], float)  # Body frame offset vector
        self.D2R = math.pi / 180  # Degrees to radians
        self.rho_w = self.rho = 1026  # Water density (kg/m³)
        self.g = 9.81  # Gravity acceleration (m/s²)

        # Reference values and current
        self.V_c = V_current  # Current water speed
        self.beta_c = beta_current * self.D2R  # Current water direction (rad)

        # Initialize state vectors
        self.nu = np.zeros(6)  # [u, v, w, p, q, r]
        self.eta = np.zeros(7)  # [x, y, z, q1, q2, q3, q0]
        self.eta[6] = 1.0  # Initialize quaternion to identity rotation

        # Initialize the AUV model
        self.name = ("SAM")
        self.L = self.l_SS  # length (m)
        self.diam = self.d_SS  # cylinder diameter (m)

        # Hydrodynamics (Fossen 2021, Section 8.4.2)
        self.S = 0.7 * self.L * self.diam  # S = 70% of rectangle L * diam
        self.a = self.L / 2  # semi-axes
        self.b = self.diam / 2

        #self.r_bg = 1e-6*np.array([0.75, 0, 0.05], float)  # CG w.r.t. to the CO, we recalculate that in calculate_cg
        self.r_bg = np.array([0.75, 0, 0.05], float)  # CG w.r.t. to the CO, we recalculate that in calculate_cg
        self.r_bb = np.array([0.75, 0, -0.06], float)  # CB w.r.t. to the CO

        # Parasitic drag coefficient CD_0, i.e. zero lift and alpha = 0
        # F_drag = 0.5 * rho * Cd * (pi * b^2)
        # F_drag = 0.5 * rho * CD_0 * S
        Cd = 0.42  # from Allen et al. (2000)
        self.CD_0 = Cd * math.pi * self.b ** 2 / self.S

        # Rigid-body mass matrix expressed in CO
        self.m_dry = self.m_SS + self.m_lcg 
        self.x_vbs = 0.0
        self.r_vbs = 0.0425 # Radius of the VBS
        self.m_vbs = self.rho_w * np.pi * self.r_vbs ** 2 * 0.0225
        self.m = self.m_dry + self.m_vbs
        self.J_total = np.zeros((3,3)) 
        self.MRB = np.zeros((6,6)) 

        # Weight and buoyancy
        self.W = self.m * self.g
        self.B = self.W # + 0.15 * self.g # NOTE: initializing the buoyancy force now with the dry weight + half the VBS filled.

        # Mass matrix including added mass
        self.M = np.zeros((6,6)) #self.MRB + self.MA
        self.Minv = np.zeros((6,6)) #np.linalg.inv(self.M)

        # Natural frequencies in roll and pitch
        self.w_roll = 0.0 
        self.w_pitch = 0.0 

        # Low-speed linear damping matrix parameters
        self.T_surge = 20  # time constant in surge (s)
        self.T_sway = 20  # time constant in sway (s)
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3  # relative damping ratio in roll
        self.zeta_pitch = 0.8  # relative damping ratio in pitch
        self.T_yaw = 1  # time constant in yaw (s)


    def init_vehicle(self):
#        # Initialize subsystems
        self.l_SS=1.5
        self.d_SS=0.19
        self.m_SS=14.9
        r_ss_cg = np.array([0.75, 0, 0.06])
        self.r_ss_cg = r_ss_cg
#        self.solid_structure = SolidStructure(
#            l_SS=1.5,
#            d_SS=0.19,
#            m_SS=14.9,
#            #J_SS_c=np.array([[10, 0, 0], [0, 15, 0], [0, 0, 20]]),
#            # Moment of inertia based on cylinder
#            J_SS_c=np.array([[0.5 * m_SS * (d_SS/2)**2, 0, 0],
#                             [0, 1/12 * m_SS * (3*(d_SS/2)**2 + l_SS**2), 0],
#                             [0, 0, 1/12 * m_SS * (3*(d_SS/2)**2 + l_SS**2)]]),
#            r_SS_c=np.array([0.75, 0, 0.1])
#        )

        # NOTE: Adjusted values
        # All values in m
        # Now the water mass for a full VBS makes more sense
        self.vbs = VariableBuoyancySystem(
            d_vbs=0.085,
            l_vbs_l=0.045,
            h_vbs=0.01,
            l_vbs_b=0.2,
            m_vbs_sh=0.1,
            r_vbs_sh_cg=np.array([0.1, 0, 0]),
            J_vbs_sh_cg=np.diag([1, 2, 3])
        )

        self.m_lcg = 2.6
#        # NOTE: Adjusted Values
#        self.lcg = LongitudinalCenterOfGravityControl(
#            l_lcg_l=0, #0.2,
#            l_lcg_r=0, #0.06,
#            l_lcg_b=0, #1.0,
#            h_lcg=0, #0.1,
#            m_lcg=0, #2.6,
#            h_lcg_dim=0, #0.1,
#            d_lcg=0 #0.1
#        )
#
#        # NOTE: Adjusted values
#        self.thruster_shaft = ThrusterShaft(
#            l_t_sh=0, #0.3,
#            r_t_sh_t=np.array([0.15, 0, -0.05])*0,
#            m_t_sh=0, #1.0,
#            J_t_sh_t=np.zeros((3,3)) #np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
#        )
#
        self.propellers = Propellers(
            n_p=2,
            l_t_p=np.zeros(2), #np.array([0.1, 0.15]),
            m_t_p=np.zeros(2), #np.array([1.5, 1.8]),
            r_t_p=[
                np.zeros(3), #np.array([0.05, 0, -0.02]),
                np.zeros(3) #np.array([0.075, 0, -0.03])
            ],
            r_t_p_sh=[
                np.zeros(3), #np.array([0.3, 0, 0]),
                np.zeros(3) #np.array([0.4, 0, 0])
            ],
            J_t_p=[
                np.zeros((3,3)), #np.array([[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0.5]]),
                np.zeros((3,3)) #np.array([[0.35, 0, 0], [0, 0.45, 0], [0, 0, 0.55]])
            ]
        )



    def dynamics(self, x, u):
        """
        Main dynamics function for integrating the complete AUV state.

        Args:
            t: Current time
            state_vector: Combined state vector [eta, nu, ksi, ksi_dot]
            signal_generator: MultiVariablePiecewiseSignal object for ksi_ddot signals

        Returns:
            state_vector_dot: Time derivative of complete state vector
        """

        # FIXME: everything right now in the body frame (eq. 8.2)
        #   But we also need everything in the global frame (eq. 8.1)

        eta = x[0:7]
        nu = x[7:13]

        self.calculate_system_state(nu, eta, u)
        self.calculate_cg(nu, u)
        self.update_inertias(nu, u)
        self.calculate_M(nu, u)
        self.calculate_C(nu, u)
        self.calculate_D(nu, u)
        self.calculate_g()
        self.calculate_tau(nu, u)

        #self.Minv = np.eye(6)
        #self.tau = np.zeros(6)
        #self.C = np.zeros((6,6))
        #self.D = np.zeros((6,6))

        print(f"g_vect: {self.g_vec}")

        nu_dot = self.Minv @ (self.tau - np.matmul(self.C,self.nu_r) - np.matmul(self.D,self.nu_r) - self.g_vec)

        eta_dot = self.eta_dynamics(eta, nu)

        x_dot = np.concatenate([eta_dot, nu_dot])

        return x_dot



    def calculate_system_state(self, x, eta, u_control):
        """
        Extract speeds etc. based on state and control inputs
        """
        nu = x
        # Extract Euler angles
        q = eta[3:7]
        self.phi, self.theta, self.psi = quaternion_to_angles(q) # NOTE: the function uses quaternions as q1, q2, q3, q0

        # Relative velocities due to current
        u, v, w, p, q, r = nu
        u_c = self.V_c * math.cos(self.beta_c - self.psi)
        v_c = self.V_c * math.sin(self.beta_c - self.psi)
        self.nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)
        self.nu_r = nu - self.nu_c

        self.U = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        self.U_r = np.linalg.norm(self.nu_r[:3])

        self.alpha = 0.0
        if abs(self.nu_r[0]) > 1e-6:
            self.alpha = math.atan2(self.nu_r[2], self.nu_r[0])

        # Update mass
        self.r_vbs = 0.0425 # Radius of the VBS
        self.x_vbs = self.calculate_vbs_position(u_control) 
        self.m_vbs = self.rho_w * np.pi * self.r_vbs ** 2 * self.x_vbs


    def calculate_cg(self, x, u):
        """
        Compute the center of gravity based on VBS and LCG position
        """

        # FIXME: To be replaced with the actual calculation
        #self.r_bg = np.array([0.75, 0, 0.06], float)  # Current center of gravity

    def update_inertias(self, x,u):
        """
        Update inertias based on VBS and LCG
        """

        self.m = self.m_dry + self.m_vbs
        # Solid structure
        # Moment of inertia of a solid elipsoid
        # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        # with b = c.
        # NOTE: m is m_dry
        Ix = (2 / 5) * self.m * self.b ** 2  # moment of inertia
        Iy = (1 / 5) * self.m * (self.a ** 2 + self.b ** 2)
        Iz = Iy

        J_ss_cg = np.diag([Ix, Iy, Iz]) # In center of gravity
        S2_r_ss_cg = skew_symmetric(self.r_ss_cg) @ skew_symmetric(self.r_ss_cg)
        J_ss_co = J_ss_cg - self.m_dry * S2_r_ss_cg

        # VBS
#        self.r_vbs = 0.0425 # Radius of the VBS
#        x_vbs = self.calculate_vbs_position(u) 
#        self.m_vbs = self.rho_w * np.pi * self.r_vbs ** 2 * x_vbs
        self.r_vbs_cg = np.array([-0.398, 0, 0.0125]) + np.array([self.x_vbs, 0, 0])# Vector for cg of the vbs to CO
        

        # Moment of inertia of a solid cylinder
        Ix_vbs = (1/2) * self.m_vbs * self.r_vbs**2
        Iy_vbs = (1/12) * self.m_vbs * (3*self.r_vbs**2 + self.x_vbs**2)
        Iz_vbs = Iy_vbs

        J_vbs_cg = np.diag([Ix_vbs, Iy_vbs, Iz_vbs])
        S2_r_vbs_cg = skew_symmetric(self.r_vbs_cg) @ skew_symmetric(self.r_vbs_cg)
        J_vbs_co = J_vbs_cg - self.m_vbs * S2_r_vbs_cg

        # FIXME: To be replaced with the actual calculation
        self.J_total = J_ss_co# + J_vbs_co

    def calculate_M(self, x, u):

        # Extract vehicle parameters
        rho = self.rho_w  # Density of water
        d_vbs = self.vbs.d_vbs  # Diameter of the VBS
        l_vbs_b = self.vbs.l_vbs_b  # Offset length of VBS

        # Extract control inputs
        x_vbs = self.calculate_vbs_position(u)

        # Rigid-body mass matrix expressed in CO
        m_water = self.m_vbs# (rho * np.pi * d_vbs ** 2 * x_vbs) / 4 
        self.m = m_water + self.m_dry
        m_diag = np.diag([self.m, self.m, self.m])

        # FIXME: This needs to be adjusted
        #   - In which frame do we compute J_total?
        #   - Does the r_bg correspond to the correct vector?
        MRB_CG = block_diag(m_diag, self.J_total)
        H_rg = Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg  # MRB expressed in the CO

        # Added moment of inertia in roll: A44 = r44 * Ix
        r44 = 0.3
        MA_44 = r44 * self.J_total[0,0]

        # FIXME: This seems to be static, so put it back into init
        # Lamb's k-factors
        e = math.sqrt(1 - (self.b / self.a) ** 2)
        alpha_0 = (2 * (1 - e ** 2) / pow(e, 3)) * (0.5 * math.log((1 + e) / (1 - e)) - e)
        beta_0 = 1 / (e ** 2) - (1 - e ** 2) / (2 * pow(e, 3)) * math.log((1 + e) / (1 - e))

        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0 / (2 - beta_0)
        k_prime = pow(e, 4) * (beta_0 - alpha_0) / (
                (2 - e ** 2) * (2 * e ** 2 - (2 - e ** 2) * (beta_0 - alpha_0)))

        # Added mass system matrix expressed in the CO
        self.MA = np.diag([self.m * k1, self.m * k2, self.m * k2, MA_44, k_prime * self.J_total[1,1], k_prime * self.J_total[1,1]])

        # Mass matrix including added mass
        self.M = self.MRB + self.MA

        self.Minv = np.linalg.inv(self.M)


    def calculate_C(self, x, u):
        """
        Calculate Corriolis Matrix
        """
        CRB = m2c(self.MRB, self.nu_r)
        CA = m2c(self.MA, self.nu_r)

        # Zero certain CA terms if originally done so
        CA[4, 0] = 0
        CA[0, 4] = 0
        CA[4, 2] = 0
        CA[2, 4] = 0
        CA[5, 0] = 0
        CA[0, 5] = 0
        CA[5, 1] = 0
        CA[1, 5] = 0

        self.C = CRB + CA

    def calculate_D(self, x, u):
        """
        Calculate damping
        """
        # Natural frequencies in roll and pitch
        self.w_roll = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
                                self.M[3][3])
        self.w_pitch = math.sqrt(self.W * (self.r_bg[2] - self.r_bb[2]) /
                                 self.M[4][4])
        # Damping
        self.D = np.diag([
            self.M[0, 0] / self.T_surge,
            self.M[1, 1] / self.T_sway,
            self.M[2, 2] / self.T_heave,
            self.M[3, 3] * 2 * self.zeta_roll * self.w_roll,
            self.M[4, 4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5, 5] / self.T_yaw
        ])
        self.D[0, 0] *= math.exp(-3 * self.U_r)
        self.D[1, 1] *= math.exp(-3 * self.U_r)

    def calculate_g(self):
        """
        Calculate gravity vector
        """
        self.W = self.m * self.g
        print(f"W: {self.W}, B: {self.B}")
#        print(f"r_bg: {self.r_bg}")
#        print(f"r_bb: {self.r_bb}")
#        print(f"r_diff: {self.r_bg - self.r_bb}")

        self.g_vec = gvect(self.W, self.B, self.theta, self.phi, self.r_bg, self.r_bb)

    def calculate_tau(self, x, u):
        """
        All external forces
        """
        # FIXME: forceLiftDrag takes forever to compute.

        #tau_liftdrag = forceLiftDrag(self.diam, self.S, self.CD_0, self.alpha, self.U_r)
        tau_crossflow = crossFlowDrag(self.L, self.diam, self.diam, self.nu_r)
        tau_prop = self.calculate_propeller_force(x, u)

        tau_liftdrag = np.zeros(6)
        #tau_crossflow = np.zeros(6)
        #tau_prop = np.zeros(6)

        self.tau = tau_liftdrag + tau_crossflow + tau_prop



    def calculate_propeller_force(self, x, u):
        """
        Calculate force and torque of the propellers
        """
        delta_s = u[2]
        delta_r = u[3]
        n_rpm = u[4:]

        # Compute propeller forces
        C_T2C = calculate_dcm(order=[2, 3], angles=[delta_s, delta_r])

        D_prop = 0.14
        t_prop = 0.1
        n_rps = n_rpm / 60   # NOTE: This assumes that rpm is the third entry
        Va = 0.944 * self.U

        KT_0 = 0.4566
        KQ_0 = 0.0700
        KT_max = 0.1798
        KQ_max = 0.0312
        Ja_max = 0.6632

        tau_prop = np.zeros(6)
        for i in range(len(n_rpm)):
            if n_rps[i] > 0:
                X_prop_i = self.rho * (D_prop ** 4) * (
                        KT_0 * abs(n_rps[i]) * n_rps[i] +
                        (KT_max - KT_0) / Ja_max * (Va / D_prop) * abs(n_rps[i])
                )
                K_prop_i = self.rho * (D_prop ** 5) * (
                        KQ_0 * abs(n_rps[i]) * n_rps[i] +
                        (KQ_max - KQ_0) / Ja_max * (Va / D_prop) * abs(n_rps[i])
                )
            else:
                X_prop_i = self.rho * (D_prop ** 4) * KT_0 * abs(n_rps[i]) * n_rps[i]
                K_prop_i = self.rho * (D_prop ** 5) * KQ_0 * abs(n_rps[i]) * n_rps[i]

            F_prop_b = C_T2C @ np.array([X_prop_i, 0, 0])
            r_prop_i = C_T2C @ self.propellers.r_t_p_sh[i] - self.r_cb
            M_prop_i = np.cross(r_prop_i, F_prop_b) + np.array([K_prop_i, 0, 0])
            tau_prop_i = np.concatenate([F_prop_b, M_prop_i])
            tau_prop += tau_prop_i

        return tau_prop


    def calculate_vbs_position(self, u):
        """
        Control input is scaled between 0 and 100. This converts it into the actual position
        s.t. we can calculate the amount of water in the VBS.
        """
        x_vbs = (u[4]/100) * self.vbs.l_vbs_l
        return x_vbs

    def eta_dynamics(self, eta, nu):
        """
        Computes the time derivative of position and quaternion orientation.

        Args:
            eta: [x, y, z, q1, q2, q3, q0] - Position and quaternion
            nu: [u, v, w, p, q, r] - Body-fixed velocities

        Returns:
            eta_dot: [ẋ, ẏ, ż, q̇0, q̇1, q̇2, q̇3]
        """
        # Extract position and quaternion
        pos = eta[0:3]
        q = eta[3:7]  # [q1, q2, q3, q0] where q0 is scalar part

        # Convert quaternion to DCM for position kinematics
        C = quaternion_to_dcm(q)

        # Position dynamics: ṗ = C * v
        pos_dot = C @ nu[0:3]

        # Quaternion dynamics: q̇ = 1/2 * Ω * q where Ω is the quaternion kinematic matrix
        omega = nu[3:6]  # Angular velocity
        Omega = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])
        q_dot = 0.5 * Omega @ q

        return np.concatenate([pos_dot, q_dot])

##############################################################################
#
# OMIDS MODEL
#
#    def calculate_vbs(self, ksi, ksi_dot, ksi_ddot):
#        """
#        Calculates the VBS contribution to the moment of inertia and its derivative, including additional terms.
#
#        Args:
#          ksi (list): Vector of time-varying parameters:
#            - ksi[0] (float): x_vbs, position of the VBS (m).
#            - ksi[1] (float): x_lcg, position of the LCG (m).
#            - ksi[2] (float): delta_e, stern plane angle (rad).
#            - ksi[3] (float): delta_r, rudder angle (rad).
#            - ksi[4:] (list): theta_rpm_i, angles of rotation of each propeller \( i \) (list of floats).
#
#          ksi_dot (list): First derivatives of time-varying parameters:
#            - ksi_dot[0] (float): x_dot_vbs, velocity of the VBS (m/s).
#            - ksi_dot[1] (float): x_dot_lcg, velocity of the LCG (m/s).
#            - ksi_dot[2] (float): delta_e_dot, rate of change of stern plane angle (rad/s).
#            - ksi_dot[3] (float): delta_r_dot, rate of change of rudder angle (rad/s).
#            - ksi_dot[4:] (list): theta_dot_rpm_i, rates of change of propeller angles (rad/s).
#
#          ksi_ddot (list): Second derivatives of time-varying parameters:
#            - ksi_ddot[0] (float): x_ddot_vbs, Acceleration of the VBS (m/s^2).
#            - ksi_ddot[1] (float): x_dot_lcg, Acceleration of the LCG (m/s^2).
#            - ksi_ddot[2] (float): delta_e_ddot, Acceleration of stern plane angle (rad/s^2).
#            - ksi_ddot[3] (float): delta_r_ddot, Acceleration of rudder angle (rad/s^2).
#            - ksi_ddot[4:] (list): theta_ddot_rpm_i, Acceleration of propeller angles (rad/s^2).
#
#        Returns:
#            dict: A dictionary containing: SI Units (Kg, m ,s)
#                - J_vbs_total (np.array): Total moment of inertia for VBS in the body frame. (3x3 matrix)
#                - J_dot_vbs_total (np.array): Time derivative of the total moment of inertia for VBS in the body frame. (3x3 matrix)
#                - J_vbs_shaft (np.array): Total moment of inertia for VBS in the body frame. (3x3 matrix)
#                - J_dot_vbs_shaft (np.array): Time derivative of the total moment of inertia for VBS in the body frame. (3x3 matrix)
#                - J_vbs_water (np.array): Total moment of inertia for VBS in the body frame. (3x3 matrix)
#                - J_dot_vbs_water (np.array): Time derivative of the water moment of inertia for VBS in the body frame. (3x3 matrix)
#                - m_vbs_w (float): Mass of VBS water.
#                - m_dot_vbs_w (float): Derivative of VBS water mass.
#                - m_ddot_vbs_w (float): Second derivative of VBS water mass.
#                - r_vbs_sh (np.array): CG position of the VBS shaft in Central frame. (3x1 vector)
#                - r_vbs_w (np.array): CG position of the VBS water in Central frame. (3x1 vector)
#                - r_dot_vbs_sh (np.array): Velocity of the VBS shaft CG in body frame. (3x1 vector)
#                - r_dot_vbs_w (np.array): Velocity of the VBS water CG in body frame. (3x1 vector)
#                - r_ddot_vbs_sh (np.array): Acceleration of the VBS shaft CG in body frame. (3x1 vector)
#                - r_ddot_vbs_w (np.array): Acceleration of the VBS water CG in body frame. (3x1 vector)
#                - h_add_vbs (np.array): Added angular momentum. (3x1 vector)
#                - h_add_vbs_dot (np.array): Derivative of the added angular momentum. (3x1 vector)
#        """
#        # Extract time-varying parameter for VBS
#        x_vbs = ksi[0]
#        x_dot_vbs = ksi_dot[0]
#        x_ddot_vbs = ksi_ddot[0]
#
#        # Extract VBS parameters from the SAM class
#        rho = self.rho_w  # Density of water
#        d_vbs = self.vbs.d_vbs  # Diameter of the VBS
#        l_vbs_b = self.vbs.l_vbs_b  # Offset length of VBS
#        h_vbs = self.vbs.h_vbs  # Vertical offset of VBS CG
#        m_vbs_sh = self.vbs.m_vbs_sh  # Shaft mass
#        J_vbs_sh_cg = self.vbs.J_vbs_sh_cg  # Shaft moment of inertia
#        r_cb = self.r_cb  # Central body vector from SAM class
#        r_vbs_sh_cg = np.array(self.vbs.r_vbs_sh_cg)  # Shaft CG position from SAM class
#
#        # --- Mass of VBS Water ---
#        m_vbs_w = (rho * np.pi * d_vbs ** 2 * x_vbs) / 4 
#        m_dot_vbs_w = (rho * np.pi * d_vbs ** 2 * x_dot_vbs) / 4
#        m_ddot_vbs_w = (rho * np.pi * d_vbs ** 2 * x_ddot_vbs) / 4
#
#        # --- Inertia of VBS Water around its CG ---
#        J1 = 0
#        J2 = J3 = (1 / 12) * m_vbs_w * ((3 / 4) * d_vbs ** 2 + x_vbs ** 2)
#        J_vbs_w_CG = np.diag([J1, J2, J3])
#
#        # --- CG Position of VBS Shaft in Central Frame ---
#        r_vbs_sh = np.array([x_vbs, 0, h_vbs]) + r_vbs_sh_cg
#        r_vbs_w = np.array([x_vbs / 2 + l_vbs_b, 0, h_vbs])
#
#        # --- CG Velocities ---
#        r_dot_vbs_sh = np.array([x_dot_vbs, 0, 0])  # Shaft velocity
#        r_dot_vbs_w = np.array([x_dot_vbs, 0, 0])  # Water velocity
#
#        # --- CG Accelerations ---
#        r_ddot_vbs_sh = np.array([x_ddot_vbs, 0, 0])  # Shaft acceleration
#        r_ddot_vbs_w = np.array([x_ddot_vbs, 0, 0])  # Water acceleration
#
#        # --- Shaft Moment of Inertia ---
#        S2_diff_sh = skew_symmetric(r_vbs_sh - r_cb) @ skew_symmetric(r_vbs_sh - r_cb)
#        J_vbs_sh_B = J_vbs_sh_cg - m_vbs_sh * S2_diff_sh
#
#        # --- Shaft Derivative ---
#        D_S2_sh = dot_skew_squared(r_vbs_sh - r_cb, r_dot_vbs_sh)
#        J_dot_vbs_sh_B = -m_vbs_sh * D_S2_sh
#
#        # --- Water Moment of Inertia ---
#        S2_diff_w = skew_symmetric(r_vbs_w - r_cb) @ skew_symmetric(r_vbs_w - r_cb)
#        J_vbs_w_B = J_vbs_w_CG - m_vbs_w * S2_diff_w
#
#        # --- Water Derivative ---
#        J_dot_vbs_w_CG = np.diag([
#            0,
#            (1 / 12) * m_dot_vbs_w * ((3 / 4) * d_vbs ** 2 + x_vbs ** 2) + (1 / 6) * m_vbs_w * x_vbs * x_dot_vbs,
#            (1 / 12) * m_dot_vbs_w * ((3 / 4) * d_vbs ** 2 + x_vbs ** 2) + (1 / 6) * m_vbs_w * x_vbs * x_dot_vbs
#        ])
#        D_S2_w = dot_skew_squared(r_vbs_w - r_cb, r_dot_vbs_w)
#        J_dot_vbs_w_B = J_dot_vbs_w_CG - m_dot_vbs_w * S2_diff_w - m_vbs_w * D_S2_w
#
#        # --- Total Moment of Inertia and Derivative ---
#        J_vbs_total = J_vbs_sh_B + J_vbs_w_B
#        J_dot_vbs_total = J_dot_vbs_sh_B + J_dot_vbs_w_B
#
#        # --- Cross Product Term ---
#        h_add_vbs = (
#                m_vbs_sh * np.cross(r_vbs_sh - r_cb, r_dot_vbs_sh)
#                + m_vbs_w * np.cross(r_vbs_w - r_cb, r_dot_vbs_w)
#        )
#
#        # --- Derivative of Cross Product Term ---
#        h_dot_add_vbs = (
#                m_vbs_sh * np.cross(r_vbs_sh - r_cb, r_ddot_vbs_sh)
#                + m_dot_vbs_w * np.cross(r_vbs_w - r_cb, r_dot_vbs_w)
#                + m_vbs_w * np.cross(r_vbs_w - r_cb, r_ddot_vbs_w)
#        )
#
#        return dict(J_vbs_total=J_vbs_total,
#                    J_dot_vbs_total=J_dot_vbs_total,
#                    J_vbs_shaft=J_vbs_sh_B,
#                    J_dot_vbs_shaft=J_vbs_sh_B,
#                    J_vbs_water=J_dot_vbs_w_B,
#                    J_dot_vbs_water=J_dot_vbs_sh_B,
#                    m_vbs_w=m_vbs_w,
#                    m_dot_vbs_w=m_dot_vbs_w,
#                    m_ddot_vbs_w=m_ddot_vbs_w,
#                    r_vbs_sh=r_vbs_sh,
#                    r_vbs_w=r_vbs_w,
#                    r_dot_vbs_sh=r_dot_vbs_sh,
#                    r_dot_vbs_w=r_dot_vbs_w,
#                    r_ddot_vbs_sh=r_ddot_vbs_sh,
#                    r_ddot_vbs_w=r_ddot_vbs_w,
#                    h_add_vbs=h_add_vbs,
#                    h_dot_add_vbs=h_dot_add_vbs)
#
#    def calculate_ss(self):
#        """
#        Calculates the Solid Structure (SS) contribution to the moment of inertia and its derivative.
#
#        Returns:
#            dict: A dictionary containing:
#                - "J_ss": Total moment of inertia for the solid structure in the body frame (3x3 matrix).
#                - "J_dot_ss": Time derivative of the total moment of inertia for the solid structure (3x3 matrix, always 0).
#                - "r_ss": CG position of solid structure in the Central frame (3x1 vector).
#
#
#        """
#        # Access Solid Structure parameters through the SolidStructure class
#        J_ss_c = self.solid_structure.J_SS_c  # Inertia tensor in the central frame
#        m_ss = self.solid_structure.m_SS  # Mass of the solid structure
#        r_ss_c = self.solid_structure.r_SS_c  # CG position of the solid structure in the central frame
#        r_cb = self.r_cb  # Central body vector from SAM class
#
#        # Calculate the skew-symmetric square matrices
#        S2_r_ss_c = skew_symmetric(r_ss_c) @ skew_symmetric(r_ss_c)
#        S2_diff = skew_symmetric(r_ss_c - r_cb) @ skew_symmetric(r_ss_c - r_cb)
#
#        # Total Moment of Inertia in Body Frame
#        J_ss = J_ss_c + m_ss * S2_r_ss_c - m_ss * S2_diff
#
#        # Derivative of Moment of Inertia (always zero for solid structure)
#        J_dot_ss = np.zeros((3, 3))
#
#        return dict(J_ss=J_ss,
#                    J_dot_ss=J_dot_ss,
#                    r_SS= r_ss_c)
#
#    def calculate_lcg(self, ksi, ksi_dot, ksi_ddot):
#        """
#        Calculates the LCG contribution to the moment of inertia, its derivative, and additional terms.
#
#        Args:
#          ksi (list): Vector of time-varying parameters:
#            - ksi[0] (float): x_vbs, position of the VBS (m).
#            - ksi[1] (float): x_lcg, position of the LCG (m).
#            - ksi[2] (float): delta_e, stern plane angle (rad).
#            - ksi[3] (float): delta_r, rudder angle (rad).
#            - ksi[4:] (list): theta_rpm_i, angles of rotation of each propeller \( i \) (list of floats).
#
#          ksi_dot (list): First derivatives of time-varying parameters:
#            - ksi_dot[0] (float): x_dot_vbs, velocity of the VBS (m/s).
#            - ksi_dot[1] (float): x_dot_lcg, velocity of the LCG (m/s).
#            - ksi_dot[2] (float): delta_e_dot, rate of change of stern plane angle (rad/s).
#            - ksi_dot[3] (float): delta_r_dot, rate of change of rudder angle (rad/s).
#            - ksi_dot[4:] (list): theta_dot_rpm_i, rates of change of propeller angles (rad/s).
#
#          ksi_ddot (list): Second derivatives of time-varying parameters:
#            - ksi_ddot[0] (float): x_ddot_vbs, Acceleration of the VBS (m/s^2).
#            - ksi_ddot[1] (float): x_dot_lcg, Acceleration of the LCG (m/s^2).
#            - ksi_ddot[2] (float): delta_e_ddot, Acceleration of stern plane angle (rad/s^2).
#            - ksi_ddot[3] (float): delta_r_ddot, Acceleration of rudder angle (rad/s^2).
#            - ksi_ddot[4:] (list): theta_ddot_rpm_i, Acceleration of propeller angles (rad/s^2).
#
#        Returns:
#            dict: A dictionary containing:
#                - "J_lcg": Total moment of inertia for LCG in the body frame (3x3 matrix).
#                - "J_dot_lcg": Time derivative of the total moment of inertia for LCG (3x3 matrix).
#                - "r_lcg_c": CG position of LCG in central frame (3x1 vector).
#                - "r_dot_lcg_c": Velocity of LCG CG in central frame (3x1 vector).
#                - "r_ddot_lcg_c": Acceleration of LCG CG in central frame (3x1 vector).
#                - "cross_term": Cross product term.
#                - "cross_term_dot": Derivative of the cross product term.
#        """
#        # Extract dynamic inputs
#        x_lcg = ksi[1]  # Position of the LCG
#        x_dot_lcg = ksi_dot[1]  # Velocity of the LCG
#        x_ddot_lcg = ksi_ddot[1]  # Acceleration of the LCG
#
#        # Extract LCG parameters
#        m_lcg = self.lcg.m_lcg  # Mass of LCG
#        l_lcg_l = self.lcg.l_lcg_l  # Length of the LCG
#        l_lcg_b = self.lcg.l_lcg_b  # Offset length of LCG
#        h_lcg = self.lcg.h_lcg  # Vertical offset of LCG CG
#        h_lcg_dim = self.lcg.h_lcg_dim  # Height of the LCG
#        d_lcg = self.lcg.d_lcg  # Width of the LCG
#        r_cb = self.r_cb  # Central body vector from SAM class
#
#        # --- Moment of Inertia in LCG CG Frame ---
#        J1 = (1 / 12) * m_lcg * (h_lcg_dim ** 2 + d_lcg ** 2)
#        J2 = (1 / 12) * m_lcg * (l_lcg_l ** 2 + h_lcg_dim ** 2)
#        J3 = (1 / 12) * m_lcg * (l_lcg_l ** 2 + d_lcg ** 2)
#        J_lcg_cg = np.diag([J1, J2, J3])
#
#        # --- CG Position of LCG in Central Frame ---
#        r_lcg_c = np.array([x_lcg + l_lcg_l / 2 + l_lcg_b, 0, h_lcg])
#
#        # --- Velocity of CG Position ---
#        r_dot_lcg_c = np.array([x_dot_lcg, 0, 0])
#
#        # --- Acceleration of CG Position ---
#        r_ddot_lcg_c = np.array([x_ddot_lcg, 0, 0])
#
#        # --- Body Frame Moment of Inertia ---
#        S2_diff = skew_symmetric(r_lcg_c - r_cb) @ skew_symmetric(r_lcg_c - r_cb)
#        J_lcg = J_lcg_cg - m_lcg * S2_diff
#
#        # --- Derivative of Moment of Inertia ---
#        D_S2 = dot_skew_squared(r_lcg_c - r_cb, r_dot_lcg_c)
#        J_dot_lcg = -m_lcg * D_S2
#
#        # --- Cross Product Term ---
#        h_add_lcg = m_lcg * np.cross(r_lcg_c - r_cb, r_dot_lcg_c)
#
#        # --- Derivative of Cross Product Term ---
#        h_dot_add_lcg = m_lcg * np.cross(r_lcg_c - r_cb, r_ddot_lcg_c)
#
#        return {
#            "J_lcg": J_lcg,
#            "J_dot_lcg": J_dot_lcg,
#            "r_lcg_c": r_lcg_c,
#            "r_dot_lcg_c": r_dot_lcg_c,
#            "r_ddot_lcg_c": r_ddot_lcg_c,
#            "h_add_lcg": h_add_lcg,
#            "h_dot_add_lcg": h_dot_add_lcg,
#        }
#
#    def calculate_thruster_shaft(self, ksi, ksi_dot, ksi_ddot):
#        """
#        Calculates the Thruster Shaft contribution to the moment of inertia, its derivative, and additional terms.
#
#        Args:
#            ksi (list): Vector of time-varying parameters:
#                - ksi[0] (float): x_vbs, position of the VBS (m).
#                - ksi[1] (float): x_lcg, position of the LCG (m).
#                - ksi[2] (float): delta_e, stern plane angle (rad).
#                - ksi[3] (float): delta_r, rudder angle (rad).
#                - ksi[4:] (list): theta_rpm_i, angles of rotation of each propeller \( i \) (list of floats).
#
#            ksi_dot (list): First derivatives of time-varying parameters:
#                - ksi_dot[0] (float): x_dot_vbs, velocity of the VBS (m/s).
#                - ksi_dot[1] (float): x_dot_lcg, velocity of the LCG (m/s).
#                - ksi_dot[2] (float): delta_e_dot, rate of change of stern plane angle (rad/s).
#                - ksi_dot[3] (float): delta_r_dot, rate of change of rudder angle (rad/s).
#                - ksi_dot[4:] (list): theta_dot_rpm_i, rates of change of propeller angles (rad/s).
#
#            ksi_ddot (list): Second derivatives of time-varying parameters:
#                - ksi_ddot[0] (float): x_ddot_vbs, acceleration of the VBS (m/s^2).
#                - ksi_ddot[1] (float): x_ddot_lcg, acceleration of the LCG (m/s^2).
#                - ksi_ddot[2] (float): delta_e_ddot, acceleration of stern plane angle (rad/s^2).
#                - ksi_ddot[3] (float): delta_r_ddot, acceleration of rudder angle (rad/s^2).
#                - ksi_ddot[4:] (list): theta_ddot_rpm_i, acceleration of propeller angles (rad/s^2).
#
#        Returns:
#            dict: A dictionary containing:
#                - "J_t_sh": Total moment of inertia for Thruster Shaft in the body frame (3x3 matrix).
#                - "J_dot_t_sh": Time derivative of the total moment of inertia for Thruster Shaft (3x3 matrix).
#                - "r_t_sh_c": CG position of the thruster shaft in the central frame (3x1 vector).
#                - "r_dot_t_sh_c": Velocity of the CG position of the thruster shaft in the central frame (3x1 vector).
#                - "r_ddot_t_sh_c": Acceleration of the CG position of the thruster shaft in the central frame (3x1 vector).
#                - "h_t_sh": Additional angular momentum of the thruster shaft in the body frame (3x1 vector).
#                - "h_dot_t_sh": Time derivative of the additional angular momentum of the thruster shaft (3x1 vector).
#        """
#        # Extract dynamic inputs for stern plane and rudder angles
#        delta_e = ksi[2]
#        delta_r = ksi[3]
#        delta_e_dot = ksi_dot[2]
#        delta_r_dot = ksi_dot[3]
#        delta_e_ddot = ksi_ddot[2]
#        delta_r_ddot = ksi_ddot[3]
#
#        # Extract thruster shaft parameters from the ThrusterShaft class
#        m_t_sh = self.thruster_shaft.m_t_sh  # Mass of the thruster shaft
#        J_t_sh_t = self.thruster_shaft.J_t_sh_t  # Moment of inertia tensor in the thruster frame
#        r_t_sh_t = self.thruster_shaft.r_t_sh_t  # Position of the thruster shaft CG in the thruster frame
#        r_cb = self.r_cb  # Central body vector
#
#        # --- Transformation Matrices ---
#        C_T2C = calculate_dcm(order=[2, 3], angles=[delta_e, delta_r])  # C_T^C
#        C_C2T = calculate_dcm(order=[3, 2], angles=[-delta_r, -delta_e])  # C_C^T
#        C_2_temp = calculate_dcm(order=[2], angles=[-delta_e])  # C_2(-delta_e)
#
#        # --- Angular Velocities ---
#        omega_tc_t = np.array([0, -delta_e_dot, 0]) + C_2_temp @ np.array([0, 0, -delta_r_dot])
#        omega_tc_c = C_T2C @ omega_tc_t
#
#        # --- Angular Accelerations ---
#        C_2_dot = dcm_derivative_single_axis(-delta_e, delta_e_dot, axis=2)
#        omega_dot_tc_t = np.array([0, -delta_e_ddot, 0]) - C_2_dot @ np.array(
#            [0, 0, -delta_r_dot]) + C_2_temp @ np.array([0, 0, -delta_r_ddot])
#        omega_dot_tc_c = C_T2C @ omega_dot_tc_t
#
#        # --- Adjust J_t_sh_t to CG ---
#        S2_r_t_sh_t = skew_symmetric(r_t_sh_t) @ skew_symmetric(r_t_sh_t)
#        J_t_sh_t_cg = J_t_sh_t + m_t_sh * S2_r_t_sh_t
#
#        # --- Compute J_prime_t_sh ---
#        J_prime_t_sh = C_T2C @ J_t_sh_t_cg @ C_C2T
#
#        # --- Compute CG Position ---
#        r_t_sh_c = C_T2C @ r_t_sh_t
#        r_dot_t_sh_c = C_T2C @ (skew_symmetric(omega_tc_t) @ r_t_sh_t)
#        r_ddot_t_sh_c = (
#                C_T2C @ (
#                    skew_symmetric(omega_tc_t) @ skew_symmetric(omega_tc_t) + skew_symmetric(omega_dot_tc_t)) @ r_t_sh_t
#        )
#        # --- Compute J_t_sh ---
#        S2_r_t_sh_c = skew_symmetric(r_t_sh_c - r_cb) @ skew_symmetric(r_t_sh_c - r_cb)
#        J_t_sh = J_prime_t_sh - m_t_sh * S2_r_t_sh_c
#
#        # --- Derivative of Moment of Inertia ---
#        D_S2 = dot_skew_squared(r_t_sh_c - r_cb, r_dot_t_sh_c)
#        J_dot_t_sh = (
#                skew_symmetric(omega_tc_c) @ J_prime_t_sh
#                - J_prime_t_sh @ skew_symmetric(omega_tc_c)
#                - m_t_sh * D_S2
#        )
#
#        # --- Additional Angular Momentum ---
#        h_t_sh = (
#                C_T2C @ J_prime_t_sh @ omega_tc_t
#                + m_t_sh * skew_symmetric(r_t_sh_c - r_cb) @ r_dot_t_sh_c
#        )
#
#        # --- Derivative of Additional Angular Momentum ---
#        h_dot_t_sh = (
#                skew_symmetric(omega_tc_c) @ C_T2C @ J_prime_t_sh @ omega_tc_t
#                + C_T2C @ J_prime_t_sh @ omega_dot_tc_t
#                + m_t_sh * skew_symmetric(r_t_sh_c - r_cb) @ r_ddot_t_sh_c
#        )
#
#        return {
#            "J_t_sh": J_t_sh,
#            "J_dot_t_sh": J_dot_t_sh,
#            "r_t_sh_c": r_t_sh_c,
#            "r_dot_t_sh_c": r_dot_t_sh_c,
#            "r_ddot_t_sh_c": r_ddot_t_sh_c,
#            "h_t_sh": h_t_sh,
#            "h_dot_t_sh": h_dot_t_sh,
#        }
#
#    def calculate_thruster_propeller(self, ksi, ksi_dot, ksi_ddot):
#        """
#        Calculates the Thruster Propeller contribution to the moment of inertia, its derivative,
#        and additional angular momentum terms, as well as intermediate results.
#
#        Args:
#            ksi (list): Vector of time-varying parameters:
#                - ksi[0] (float): x_vbs, position of the VBS (m).
#                - ksi[1] (float): x_lcg, position of the LCG (m).
#                - ksi[2] (float): delta_e, stern plane angle (rad).
#                - ksi[3] (float): delta_r, rudder angle (rad).
#                - ksi[4:] (list): theta_rpm_i, angles of rotation of each propeller \( i \) (list of floats).
#
#            ksi_dot (list): First derivatives of time-varying parameters:
#                - ksi_dot[0] (float): x_dot_vbs, velocity of the VBS (m/s).
#                - ksi_dot[1] (float): x_dot_lcg, velocity of the LCG (m/s).
#                - ksi_dot[2] (float): delta_e_dot, rate of change of stern plane angle (rad/s).
#                - ksi_dot[3] (float): delta_r_dot, rate of change of rudder angle (rad/s).
#                - ksi_dot[4:] (list): theta_dot_rpm_i, rates of change of propeller angles (rad/s).
#
#            ksi_ddot (list): Second derivatives of time-varying parameters:
#                - ksi_ddot[0] (float): x_ddot_vbs, acceleration of the VBS (m/s^2).
#                - ksi_ddot[1] (float): x_ddot_lcg, acceleration of the LCG (m/s^2).
#                - ksi_ddot[2] (float): delta_e_ddot, acceleration of stern plane angle (rad/s^2).
#                - ksi_ddot[3] (float): delta_r_ddot, acceleration of rudder angle (rad/s^2).
#                - ksi_ddot[4:] (list): theta_ddot_rpm_i, acceleration of propeller angles (rad/s^2).
#
#        Returns:
#            dict: A dictionary containing:
#                - "J_tp_total": Total moment of inertia for Thruster Propellers in the body frame (3x3 matrix).
#                - "J_dot_tp_total": Total time derivative of the moment of inertia for Thruster Propellers (3x3 matrix).
#                - "h_tp_total": Total angular momentum of Thruster Propellers in the body frame (3x1 vector).
#                - "h_dot_tp_total": Total time derivative of angular momentum of Thruster Propellers (3x1 vector).
#                - "h_tp_list": List of angular momenta of each propeller in the body frame.
#                - "h_dot_tp_list": List of time derivatives of angular momenta of each propeller in the body frame.
#                - "J_tp_individual": List of moment of inertia matrices for each propeller (list of 3x3 matrices).
#                - "J_dot_tp_individual": List of time derivatives of moment of inertia matrices for each propeller (list of 3x3 matrices).
#                - "positions": List of CG positions for each propeller in the body frame.
#                - "velocities": List of CG velocities for each propeller in the body frame.
#                - "accelerations": List of CG accelerations for each propeller in the body frame.
#                - "omega_pc_p": List of angular velocities of each propeller in the propeller frame.
#                - "omega_dot_pc_p": List of angular accelerations of each propeller in the propeller frame.
#                - "dcms": List of DCMs for each propeller frame relative to the body frame.
#        """
#        # Extract dynamic inputs
#        delta_e = ksi[2]  # Stern plane angle
#        delta_r = ksi[3]  # Rudder angle
#        theta_rpm_i = ksi[4:]  # List of propeller rotation angles
#        delta_e_dot = ksi_dot[2]  # Rate of change of stern plane angle
#        delta_r_dot = ksi_dot[3]  # Rate of change of rudder angle
#        theta_dot_rpm_i = ksi_dot[4:]  # List of rates of change of propeller angles
#        delta_e_ddot = ksi_ddot[2]  # Acceleration of stern plane angle
#        delta_r_ddot = ksi_ddot[3]  # Acceleration of rudder angle
#        theta_ddot_rpm_i = ksi_ddot[4:]  # List of accelerations of propeller angles
#
#        # Extract propeller parameters
#        n_p = self.propellers.n_p  # Number of propellers
#        m_tp = self.propellers.m_t_p  # List of masses for each propeller
#        J_tp_p = self.propellers.J_t_p  # List of inertia tensors for each propeller
#        r_tp_p = self.propellers.r_t_p  # List of CG positions for each propeller
#        r_tp_sh = self.propellers.r_t_p_sh  # List of positions of propeller on shaft
#        r_cb = self.r_cb  # Central body vector
#
#        # Initialize total contributions
#        J_tp_total = np.zeros((3, 3))
#        J_dot_tp_total = np.zeros((3, 3))
#        h_tp_total = np.zeros(3)
#        h_dot_tp_total = np.zeros(3)
#        h_tp_list = []
#        h_dot_tp_list = []
#        J_tp_individual = []
#        J_dot_tp_individual = []
#        positions = []
#        velocities = []
#        accelerations = []
#        omega_pc_p_list = []
#        omega_dot_pc_p_list = []
#        dcms = []
#
#        # Precompute angular velocities and accelerations for the central frame
#        C_T2C = calculate_dcm(order=[2, 3], angles=[delta_e, delta_r])  # C_T^C
#        omega_tc_t = np.array([0, -delta_e_dot, 0]) + calculate_dcm(order=[2], angles=[-delta_e]) @ np.array(
#            [0, 0, -delta_r_dot])
#        omega_dot_tc_t = (
#                np.array([0, -delta_e_ddot, 0])
#                - dcm_derivative_single_axis(-delta_e, delta_e_dot, axis=2) @ np.array([0, 0, -delta_r_dot])
#                + calculate_dcm(order=[2], angles=[-delta_e]) @ np.array([0, 0, -delta_r_ddot])
#        )
#
#        for i in range(n_p):
#            # Extract propeller-specific parameters
#            m_tp_i = m_tp[i]
#            J_tp_p_i = J_tp_p[i]
#            r_tp_p_i = r_tp_p[i]
#            r_tp_sh_i = r_tp_sh[i]
#            theta_rpm = theta_rpm_i[i]
#            theta_dot_rpm = theta_dot_rpm_i[i]
#            theta_ddot_rpm = theta_ddot_rpm_i[i]
#
#            # --- Transformation Matrices ---
#            C_P2T = calculate_dcm(order=[1], angles=[-theta_rpm])
#            C_T2P = C_P2T.T
#            C_P2C = C_T2C @ C_P2T
#            C_C2P = C_P2C.T
#
#            # Store DCM
#            dcms.append(C_P2C)
#
#            # --- Angular Velocities ---
#            omega_pc_p = np.array([theta_dot_rpm, 0, 0]) + C_T2P @ omega_tc_t  # Angular velocity in propeller frame
#            omega_dot_pc_p = (
#                    np.array([theta_ddot_rpm, 0, 0])
#                    + dcm_derivative_single_axis(theta_rpm, theta_dot_rpm, axis=1) @ omega_tc_t
#                    + C_T2P @ omega_dot_tc_t
#            )
#            omega_pc_c = C_P2C @ omega_pc_p
#            omega_dot_pc_c = C_P2C @ omega_dot_pc_p
#
#            # Store angular velocities and accelerations
#            omega_pc_p_list.append(omega_pc_p)
#            omega_dot_pc_p_list.append(omega_dot_pc_p)
#
#            # --- CG Positions ---
#            r_tp_c = C_T2C @ r_tp_sh_i + C_P2C @ r_tp_p_i
#            positions.append(r_tp_c)
#
#            # --- CG Velocities ---
#            r_dot_tp_c = (
#                    C_T2C @ skew_symmetric(omega_tc_t) @ r_tp_sh_i
#                    + C_P2C @ skew_symmetric(omega_pc_p) @ r_tp_p_i
#            )
#            velocities.append(r_dot_tp_c)
#
#            # --- CG Accelerations ---
#            r_ddot_tp_c = (
#                    C_T2C @ (skew_symmetric(omega_dot_tc_t) @ r_tp_sh_i + skew_symmetric(omega_tc_t) @ skew_symmetric(
#                omega_tc_t) @ r_tp_sh_i)
#                    + C_P2C @ (skew_symmetric(omega_dot_pc_p) @ r_tp_p_i + skew_symmetric(
#                omega_pc_p) @ skew_symmetric(omega_pc_p) @ r_tp_p_i)
#            )
#            accelerations.append(r_ddot_tp_c)
#
#            # --- Adjust J_tp_p to CG ---
#            S2_r_tp_p = skew_symmetric(r_tp_p_i) @ skew_symmetric(r_tp_p_i)
#            J_tp_p_cg = J_tp_p_i + m_tp_i * S2_r_tp_p
#
#            # --- Compute J_prime_tp ---
#            J_prime_tp = C_P2C @ J_tp_p_cg @ C_C2P
#
#            # --- Compute J_tp ---
#            S2_r_tp_c = skew_symmetric(r_tp_c - r_cb) @ skew_symmetric(r_tp_c - r_cb)
#            J_tp = J_prime_tp - m_tp_i * S2_r_tp_c
#            J_tp_individual.append(J_tp)
#
#            # --- Derivative of Moment of Inertia ---
#            D_S2 = dot_skew_squared(r_tp_c - r_cb, r_dot_tp_c)
#            J_dot_tp = (
#                    skew_symmetric(omega_pc_c) @ J_prime_tp
#                    - J_prime_tp @ skew_symmetric(omega_pc_c)
#                    - m_tp_i * D_S2
#            )
#            J_dot_tp_individual.append(J_dot_tp)
#
#            # --- Additional Angular Momentum ---
#            h_tp = C_P2C @ J_tp_p_cg @ omega_pc_p + m_tp_i * skew_symmetric(r_tp_c - r_cb) @ r_dot_tp_c
#            h_tp_list.append(h_tp)
#
#            # --- Derivative of Additional Angular Momentum ---
#            h_dot_tp = (
#                    skew_symmetric(omega_pc_c) @ C_P2C @ J_tp_p_cg @ omega_pc_p
#                    + C_P2C @ J_tp_p_cg @ omega_dot_pc_p
#                    + m_tp_i * skew_symmetric(r_tp_c - r_cb) @ r_ddot_tp_c
#            )
#            h_dot_tp_list.append(h_dot_tp)
#
#            # Accumulate total contributions
#            J_tp_total += J_tp
#            J_dot_tp_total += J_dot_tp
#            h_tp_total += h_tp
#            h_dot_tp_total += h_dot_tp
#
#        return {
#            "J_tp_total": J_tp_total,
#            "J_dot_tp_total": J_dot_tp_total,
#            "J_tp_individual": J_tp_individual,
#            "J_dot_tp_individual": J_dot_tp_individual,
#            "h_tp_total": h_tp_total,
#            "h_dot_tp_total": h_dot_tp_total,
#            "h_tp_list": h_tp_list,
#            "h_dot_tp_list": h_dot_tp_list,
#            "positions": positions,
#            "velocities": velocities,
#            "accelerations": accelerations,
#            "omega_pc_p": omega_pc_p_list,
#            "omega_dot_pc_p": omega_dot_pc_p_list,
#            "dcms": dcms,
#        }
#
#    def calculate_center_of_gravity_and_dynamics(self, ksi, ksi_dot, ksi_ddot):
#        """
#        Calculates the Center of Gravity (CG), its derivatives, total angular momentum, total moment of inertia,
#        and related dynamics for the SAM AUV.
#
#        Args:
#            ksi (list): Vector of time-varying parameters.
#            ksi_dot (list): First derivatives of time-varying parameters.
#            ksi_ddot (list): Second derivatives of time-varying parameters.
#
#        Returns:
#            dict: A dictionary containing:
#                - r_BG (np.array): CG position in the body frame (3x1 vector).
#                - r_dot_BG (np.array): Velocity of the CG in the body frame (3x1 vector).
#                - r_ddot_BG (np.array): Acceleration of the CG in the body frame (3x1 vector).
#                - J_total (np.array): Total moment of inertia in the body frame (3x3 matrix).
#                - J_dot_total (np.array): Time derivative of the total moment of inertia in the body frame (3x3 matrix).
#                - h_add_total (np.array): Total additional angular momentum in the body frame (3x1 vector).
#                - h_dot_add_total (np.array): Time derivative of the total additional angular momentum (3x1 vector).
#                - position_contributions (np.array): Matrix where each column shows CG position contributions from each part.
#                - velocity_contributions (np.array): Matrix where each column shows CG velocity contributions from each part.
#                - acceleration_contributions (np.array): Matrix where each column shows CG acceleration contributions from each part.
#                - mass_contributions (np.array): Array where each element shows mass contributions from each part.
#        """
#        # Retrieve individual contributions
#        vbs = self.calculate_vbs(ksi, ksi_dot, ksi_ddot)
#        ss = self.calculate_ss()
#        lcg = self.calculate_lcg(ksi, ksi_dot, ksi_ddot)
#        thruster_shaft = self.calculate_thruster_shaft(ksi, ksi_dot, ksi_ddot)
#        propeller = self.calculate_thruster_propeller(ksi, ksi_dot, ksi_ddot)
#
#        # Extract parameters from each part
#        # Solid Structure
#        m_ss = self.solid_structure.m_SS
#        r_ss = ss["r_SS"]
#        J_ss = ss["J_ss"]
#        J_dot_ss = ss["J_dot_ss"]
#
#        # LCG
#        m_lcg = self.lcg.m_lcg
#        r_lcg_c = lcg["r_lcg_c"]
#        r_dot_lcg_c = lcg["r_dot_lcg_c"]
#        r_ddot_lcg_c = lcg["r_ddot_lcg_c"]
#        J_lcg = lcg["J_lcg"]
#        J_dot_lcg = lcg["J_dot_lcg"]
#        h_add_lcg = lcg["h_add_lcg"]
#        h_dot_add_lcg = lcg["h_dot_add_lcg"]
#
#        # Thruster Shaft
#        m_t_sh = self.thruster_shaft.m_t_sh
#        r_t_sh_c = thruster_shaft["r_t_sh_c"]
#        r_dot_t_sh_c = thruster_shaft["r_dot_t_sh_c"]
#        r_ddot_t_sh_c = thruster_shaft["r_ddot_t_sh_c"]
#        J_t_sh = thruster_shaft["J_t_sh"]
#        J_dot_t_sh = thruster_shaft["J_dot_t_sh"]
#        h_t_sh = thruster_shaft["h_t_sh"]
#        h_dot_t_sh = thruster_shaft["h_dot_t_sh"]
#
#        # Propellers
#        m_t_p = self.propellers.m_t_p
#        r_t_p_list = propeller["positions"]
#        r_dot_t_p_list = propeller["velocities"]
#        r_ddot_t_p_list = propeller["accelerations"]
#        J_tp_total = propeller["J_tp_total"]
#        J_dot_tp_total = propeller["J_dot_tp_total"]
#        h_tp_total = propeller["h_tp_total"]
#        h_dot_tp_total = propeller["h_dot_tp_total"]
#
#        # VBS
#        m_vbs_sh = self.vbs.m_vbs_sh
#        m_vbs_w = vbs["m_vbs_w"]
#        m_dot_vbs_w = vbs["m_dot_vbs_w"]
#        m_ddot_vbs_w = vbs["m_ddot_vbs_w"]
#        r_vbs_sh = vbs["r_vbs_sh"]
#        r_vbs_w = vbs["r_vbs_w"]
#        r_dot_vbs_sh = vbs["r_dot_vbs_sh"]
#        r_dot_vbs_w = vbs["r_dot_vbs_w"]
#        r_ddot_vbs_sh = vbs["r_ddot_vbs_sh"]
#        r_ddot_vbs_w = vbs["r_ddot_vbs_w"]
#        J_vbs_total = vbs["J_vbs_total"]
#        J_dot_vbs_total = vbs["J_dot_vbs_total"]
#        h_add_vbs = vbs["h_add_vbs"]
#        h_dot_add_vbs = vbs["h_dot_add_vbs"]
#
#        # Central body vector
#        r_cb = self.r_cb
#
#        # Step 2: Calculate \( N \), \( \dot{N} \), \( \ddot{N} \), \( D \), \( \dot{D} \)
#        N = (
#                m_ss * r_ss
#                + m_lcg * r_lcg_c
#                + m_t_sh * r_t_sh_c
#                + np.sum([m * r for m, r in zip(m_t_p, r_t_p_list)], axis=0)
#                + m_vbs_sh * r_vbs_sh
#                + m_vbs_w * r_vbs_w
#        )
#        N_dot = (
#                m_lcg * r_dot_lcg_c
#                + m_t_sh * r_dot_t_sh_c
#                + np.sum([m * r_dot for m, r_dot in zip(m_t_p, r_dot_t_p_list)], axis=0)
#                + m_vbs_sh * r_dot_vbs_sh
#                + m_dot_vbs_w * r_vbs_w
#                + m_vbs_w * r_dot_vbs_w
#        )
#        N_ddot = (
#                m_lcg * r_ddot_lcg_c
#                + m_t_sh * r_ddot_t_sh_c
#                + np.sum([m * r_ddot for m, r_ddot in zip(m_t_p, r_ddot_t_p_list)], axis=0)
#                + m_vbs_sh * r_ddot_vbs_sh
#                + 2 * m_dot_vbs_w * r_dot_vbs_w
#                + m_ddot_vbs_w * r_vbs_w
#                + m_vbs_w * r_ddot_vbs_w
#        )
#        D = m_ss + m_lcg + m_t_sh + np.sum(m_t_p) + m_vbs_sh + m_vbs_w
#        D_dot = m_dot_vbs_w
#        D_ddot = m_ddot_vbs_w
#
#        # Step 3: Calculate \( r_{BG} \), \( \dot{r}_{BG} \), and \( \ddot{r}_{BG} \)
#        r_BG = N / D - r_cb
#        r_dot_BG = (N_dot * D - N * D_dot) / D ** 2
#        r_ddot_BG = (N_ddot * D ** 2 - N_dot * D * D_dot - N * D_ddot + 2 * N * (D_dot ** 2)) / D ** 3
#
#        # Total J and J_dot
#        J_total = J_ss + J_lcg + J_t_sh + J_tp_total + J_vbs_total
#        J_dot_total = J_dot_ss + J_dot_lcg + J_dot_t_sh + J_dot_tp_total + J_dot_vbs_total
#
#        # Total additional angular momentum and its derivative
#        h_add_total = h_add_lcg + h_t_sh + h_tp_total + h_add_vbs
#        h_dot_add_total = h_dot_add_lcg + h_dot_t_sh + h_dot_tp_total + h_dot_add_vbs
#
#        # Step 4: Organize results
#        position_contributions = np.array(
#            [r_ss, r_lcg_c, r_t_sh_c, *r_t_p_list, r_vbs_sh, r_vbs_w]
#        ).T  # Removed `- r_cb`
#        velocity_contributions = np.array(
#            [np.zeros(3), r_dot_lcg_c, r_dot_t_sh_c, *r_dot_t_p_list, r_dot_vbs_sh, r_dot_vbs_w]
#        ).T
#        acceleration_contributions = np.array(
#            [np.zeros(3), r_ddot_lcg_c, r_ddot_t_sh_c, *r_ddot_t_p_list, r_ddot_vbs_sh, r_ddot_vbs_w]
#        ).T
#        mass_contributions = np.array([m_ss, m_lcg, m_t_sh, *m_t_p, m_vbs_sh, m_vbs_w])
#
#        return {
#            "r_BG": r_BG,
#            "r_dot_BG": r_dot_BG,
#            "r_ddot_BG": r_ddot_BG,
#            "J_total": J_total,
#            "J_dot_total": J_dot_total,
#            "h_add_total": h_add_total,
#            "h_dot_add_total": h_dot_add_total,
#            "position_contributions": position_contributions,
#            "velocity_contributions": velocity_contributions,
#            "acceleration_contributions": acceleration_contributions,
#            "mass_contributions": mass_contributions,
#        }
#
