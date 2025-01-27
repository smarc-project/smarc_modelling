#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM.py:

   Class for the SAM (Small and Affordable Maritime) cylinder-shaped autonomous underwater vehicle (AUV),
   designed for agile hydrobatic maneuvers, including obstacle avoidance, inspections, docking, and under-ice operations.
   The SAM AUV is controlled using counter-rotating propellers, a thrust vectoring system, a variable buoyancy system (VBS),
   and adjustable battery packs for center of gravity (c.g.) control. It is equipped with sensors such as IMU, DVL, GPS, and sonar.

   The length of the AUV is 1.5 m, the cylinder diameter is 19 cm, and the mass of the vehicle is 17 kg.
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

Methods:

    [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime ) returns
        nu[k+1] and u_actual[k+1] using Variable RK method(Possibility for other Methods). The control input is:

    FIXME: That's not correct anymore
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

References:

    Bhat, S., Panteli, C., Stenius, I., & Dimarogonas, D. V. (2023). Nonlinear model predictive control for hydrobatic AUVs:
        Experiments with the SAM vehicle. Journal of Field Robotics, 40(7), 1840-1859. doi:10.1002/rob.22218.

    T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and Motion Control. 2nd Edition, Wiley.
        URL: www.fossen.biz/wiley


Author:     Omid Mirzaeedodangeh

Refactored: David Doerner
"""

import numpy as np
import math
from scipy.linalg import block_diag
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

    Vectors follow Tedrake's monogram:
    https://manipulation.csail.mit.edu/pick.html#monogram
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
        self.x_vbs_dot_max = 10 # FIXME: This is an estimate. Need to adjust, since the speed is given in mm/s, but we control on percentages right now. Maximum extension speed (m/s)


class LongitudinalCenterOfGravityControl:
    """
    Represents the Longitudinal Center of Gravity Control (LCG) of the SAM AUV.

    Attributes:
        l_lcg_l: Length of the LCG structure along the x-axis (m).
        l_lcg_r: Maximum position of the LCG in the x-direction (m).
        l_lcg_b: Additional offset length along the x-axis (m).
        h_lcg: Vertical offset of the CG along the z-axis relative to the central frame (m).
        m_lcg: Mass of the LCG (kg).
        h_lcg_dim: Height of the LCG structure (m).
        d_lcg: Width of the LCG structure (m).
    """

    def __init__(self, l_lcg_l, l_lcg_r, l_lcg_b, h_lcg, m_lcg, h_lcg_dim, d_lcg):
        # Physical parameters
        self.l_lcg_l = l_lcg_l  # Length of LCG structure (m)
        self.l_lcg_r = l_lcg_r  # Maximum x-direction position (m)
        self.l_lcg_b = l_lcg_b  # Additional x-axis offset (m)
        self.h_lcg = h_lcg  # Vertical CG offset (m)
        self.m_lcg = m_lcg  # Mass of LCG (kg)
        self.h_lcg_dim = h_lcg_dim  # Height of LCG structure (m)
        self.d_lcg = d_lcg  # Width of LCG structure (m)

        # Motion bounds
        self.x_lcg_min = 0  # Minimum LCG position (m)
        self.x_lcg_max = l_lcg_r  # Maximum LCG position (m)
        self.x_lcg_dot_min = -0.1  # Maximum retraction speed (m/s)
        self.x_lcg_dot_max = 15  # FIXME: This is an estimate. Need to adjust, since the speed is given in mm/s, but we control on percentages right now. Maximum extension speed (m/s)

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
class SimpleSAM():
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
        self.p_OC_O = np.array([-0.75, 0, 0.06], float)  # Measurement frame C in CO (O)
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
        self.name = ("SAM")
        self.L = self.l_SS  # length (m)
        self.diam = self.d_SS  # cylinder diameter (m)

        # Hydrodynamics (Fossen 2021, Section 8.4.2)
        self.a = self.L / 2  # semi-axes
        self.b = self.diam / 2

        self.p_OG_O = np.array([0., 0, 0.12], float)  # CG w.r.t. to the CO, we
                                                        # recalculate that in calculate_cg
        self.p_OB_O = np.array([0., 0, 0], float)  # CB w.r.t. to the CO

        # Rigid-body mass matrix expressed in CO
        self.x_vbs = 0.0
        self.r_vbs = 0.0425 # Radius of the VBS
        self.m_vbs = self.rho_w * np.pi * self.r_vbs ** 2 * 0.0225
        self.m = self.m_ss + self.m_vbs + self.m_lcg
        self.J_total = np.zeros((3,3)) 
        self.MRB = np.zeros((6,6)) 
        self.MA = np.zeros((6,6)) 
        self.M = np.zeros((6,6)) 
        self.Minv = np.zeros((6,6)) 

        # Weight and buoyancy
        self.W = self.m * self.g
        self.B = self.W # NOTE: Init buoyancy as dry mass + half the VBS

        # Mass matrix including added mass
        self.M = np.zeros((6,6)) #self.MRB + self.MA
        self.Minv = np.zeros((6,6)) #np.linalg.inv(self.M)


    def init_vehicle(self):
        """
        Initialize all subsystems based on their respective parameters
        """

#        # Initialize subsystems
        self.l_SS=1.5
        self.d_SS=0.19
        self.m_ss=14.9
        p_CSSg_O = np.array([0.74, 0, 0.06]) 
        self.p_CSSg_O = p_CSSg_O
        self.p_OSsg_O = self.p_OC_O + self.p_CSSg_O
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
        p_CVbs_O = np.array([0.404, 0, 0.0125]) 
        self.p_OVbs_O = self.p_OC_O + p_CVbs_O # FIXME: Check this how it goes into the CG calculation of the VBS. It changes with x_vbs, so you might want to adjust it as well.
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
        self.l_lcg_l = 0.223
        p_CLcgpos_O = np.array([0.608+self.l_lcg_l/2, 0, 0.130]) # "Beginning" of the LCG in C frame. Mass moves from here
        self.p_OLcgPos_O = self.p_OC_O + p_CLcgpos_O
        self.h_lcg_dim = 0.08
        # NOTE: Adjusted Values
        self.lcg = LongitudinalCenterOfGravityControl(
            l_lcg_l=0.2,
            l_lcg_r=0, #0.06,
            l_lcg_b=0, #1.0,
            h_lcg=0, #0.1,
            m_lcg=self.m_lcg,
            h_lcg_dim=self.h_lcg_dim, #0.1,
            d_lcg=0 #0.1
        )
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
                np.array([0.03, 0, 0]),
                np.array([0.04, 0, 0])
            ],
            J_t_p=[
                np.zeros((3,3)), #np.array([[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0.5]]),
                np.zeros((3,3)) #np.array([[0.35, 0, 0], [0, 0.45, 0], [0, 0, 0.55]])
            ]
        )



    def dynamics(self, x, u_ref):
        """
        Main dynamics function for integrating the complete AUV state.

        Args:
            t: Current time
            state_vector: Combined state vector [eta, nu, ksi, ksi_dot]
            signal_generator: MultiVariablePiecewiseSignal object for ksi_ddot signals

            u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]

        Returns:
            state_vector_dot: Time derivative of complete state vector
        """
        eta = x[0:7]
        nu = x[7:13]
        u = x[13:19]

        u = self.bound_actuators(u)
        u_ref = self.bound_actuators(u_ref)

        self.calculate_system_state(nu, eta, u)
        self.calculate_cg()
        self.update_inertias()
        self.calculate_M()
        self.calculate_C()
        self.calculate_D()
        self.calculate_g()
        self.calculate_tau(u)

        nu_dot = self.Minv @ (self.tau - np.matmul(self.C,self.nu_r) - np.matmul(self.D,self.nu_r) - self.g_vec)
        eta_dot = self.eta_dynamics(eta, nu)
        u_dot = self.actuator_dynamics(u, u_ref)
        x_dot = np.concatenate([eta_dot, nu_dot, u_dot])

        return x_dot

    def bound_actuators(self, u):
        """
        Enforce actuation limits on each actuator.
        """
        u_bound = np.copy(u)

        # NOTE: We control based on percentages right now.
        #   If we want to send something different, we have to adjust here.
        if u[0] > 100: #self.vbs.x_vbs_max:
            u_bound[0] = 100 #self.vbs.x_vbs_max
        elif u[0] < 0: #self.vbs.x_vbs_min:
            u_bound[0] = 0 #self.vbs.x_vbs_min
        else:
            u_bound[0] = u[0]

        if u[1] > 100:
            u_bound[1] = 100
        elif u[1] < 0:
            u_bound[1] = 0
        else:
            u_bound[1] = u[1]

        # FIXME: Add the remaining actuator limits
        # FIXME: call them as variable

        return u_bound


    def calculate_system_state(self, x, eta, u_control):
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

        # Update actuators
        self.x_vbs = self.calculate_vbs_position(u_control) 
        self.p_OLcg_O = self.calculate_lcg_position(u_control)

        # Update mass
        self.m_vbs = self.rho_w * np.pi * self.r_vbs ** 2 * self.x_vbs
        self.m = self.m_ss + self.m_vbs + self.m_lcg


    def calculate_cg(self):
        """
        Compute the center of gravity based on VBS and LCG position
        """
        self.p_OG_O = (self.m_ss/self.m) * self.p_OSsg_O \
                    + (self.m_vbs/self.m) * self.p_OVbs_O \
                    + (self.m_lcg/self.m) * self.p_OLcg_O


    def update_inertias(self):
        """
        Update inertias based on VBS and LCG
        Note: The propellers add more torque rather than momentum by moving.
            The exception would be steering, but that's complex and will change
            in the next iteration of SAM.
        """

        # Solid structure
        # Moment of inertia of a solid elipsoid
        # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        # with b = c.
        Ix = (2 / 5) * self.m_ss * self.b ** 2  # moment of inertia
        Iy = (1 / 5) * self.m_ss * (self.a ** 2 + self.b ** 2)
        Iz = Iy

        J_ss_cg = np.diag([Ix, Iy, Iz]) # In center of gravity
        S2_p_OSsg_O = skew_symmetric(self.p_OSsg_O) @ skew_symmetric(self.p_OSsg_O)
        J_ss_co = J_ss_cg - self.m_ss * S2_p_OSsg_O

        # VBS
        # Moment of inertia of a solid cylinder
        Ix_vbs = (1/2) * self.m_vbs * self.r_vbs**2
        Iy_vbs = (1/12) * self.m_vbs * (3*self.r_vbs**2 + self.x_vbs**2)
        Iz_vbs = Iy_vbs

        J_vbs_cg = np.diag([Ix_vbs, Iy_vbs, Iz_vbs])
        S2_r_vbs_cg = skew_symmetric(self.p_OVbs_O) @ skew_symmetric(self.p_OVbs_O)
        J_vbs_co = J_vbs_cg - self.m_vbs * S2_r_vbs_cg

        # LCG
        # Moment of inertia of a solid cylinder
        Ix_lcg = (1/2) * self.m_lcg * (self.h_lcg_dim/2)**2
        Iy_lcg = (1/12) * self.m_lcg* (3*(self.h_lcg_dim/2)**2 + self.l_lcg_l**2)
        Iz_lcg = Iy_lcg

        J_lcg_cg = np.diag([Ix_lcg, Iy_lcg, Iz_lcg])
        S2_r_lcg_cg = skew_symmetric(self.p_OLcg_O) @ skew_symmetric(self.p_OLcg_O)
        J_lcg_co = J_lcg_cg - self.m_lcg * S2_r_lcg_cg

        self.J_total = J_ss_co + J_vbs_co + J_lcg_co


    def calculate_M(self):
        """
        Calculated the mass matrix M
        """

        # Rigid-body mass matrix expressed in CO
        m_diag = np.diag([self.m, self.m, self.m])

        # Rigid-body mass matrix with total inertia in CO
        MRB_CO = block_diag(m_diag, self.J_total)
        self.MRB = MRB_CO

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
        self.MA = np.diag([self.m * k1,
                           self.m * k2,
                           self.m * k2,
                           MA_44,
                           k_prime * self.J_total[1,1],
                           k_prime * self.J_total[1,1]])

        # Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)


    def calculate_C(self):
        """
        Calculate Corriolis Matrix
        """
        CRB = m2c(self.MRB, self.nu_r)
        CA = m2c(self.MA, self.nu_r)

        CA[4, 0] = 0
        CA[0, 4] = 0
        CA[4, 2] = 0
        CA[2, 4] = 0
        CA[5, 0] = 0
        CA[0, 5] = 0
        CA[5, 1] = 0
        CA[1, 5] = 0

        self.C = CRB + CA

    def calculate_D(self):
        """
        Calculate damping
        """
        # Damping matrix based on Bhat 2021
        # Parameters from smarc_advanced_controllers mpc_inverted_pendulum...
        # NOTE: These need to be identified properly
        # Damping coefficients
        Xuu = 3 #100     # x-damping
        Yvv = 50    # y-damping
        Zww = 50    # z-damping
        Kpp = 40    # Roll damping
        Mqq = 200    # Pitch damping
        Nrr = 10    # Yaw damping

        # Center of effort -> where the thrust force acts?
        x_cp = 0.1
        y_cp = 0
        z_cp = 0

        self.D = np.zeros((6,6))

        # Nonlinear damping
        self.D[0,0] = Xuu * np.abs(self.nu_r[0])
        self.D[1,1] = Yvv * np.abs(self.nu_r[1])
        self.D[2,2] = Zww * np.abs(self.nu_r[2])
        self.D[3,3] = Kpp * np.abs(self.nu_r[3])
        self.D[4,4] = Mqq * np.abs(self.nu_r[4])
        self.D[5,5] = Nrr * np.abs(self.nu_r[5])

        # Cross couplings
        self.D[4,0] = z_cp * Xuu * np.abs(self.nu_r[0])
        self.D[5,0] = -y_cp * Xuu * np.abs(self.nu_r[0])
        self.D[3,1] = -z_cp * Yvv * np.abs(self.nu_r[1])
        self.D[5,1] = x_cp * Yvv * np.abs(self.nu_r[1])
        self.D[3,2] = y_cp * Zww * np.abs(self.nu_r[2])
        self.D[4,2] = -x_cp * Zww * np.abs(self.nu_r[2])

    def calculate_g(self):
        """
        Calculate gravity vector
        """
        self.W = self.m * self.g
        self.g_vec = gvect(self.W, self.B, self.theta, self.phi, self.p_OG_O, self.p_OB_O)


    def calculate_tau(self, u):
        """
        All external forces

        Note: We use a non-diagonal damping matrix, that takes forceLiftDrag
            and the crossFlowDrag, i.e. the cross-couplings in the damping already
            into account. If you use a diagonal matrix, you have to add these
            forces here, as shown in the commented code below:

            tau_liftdrag = forceLiftDrag(self.diam, self.S, self.CD_0, self.alpha, self.nu)
            tau_crossflow = crossFlowDrag(self.L, self.diam, self.diam, self.nu_r)
        """
        tau_prop = self.calculate_propeller_force(u)
        self.tau = tau_prop


    def calculate_propeller_force(self, u):
        """
        Calculate force and torque of the propellers
        u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        Azimuth Thrusters: Fossen 2021, ch.9.4.2
        """
        delta_s = u[2]
        delta_r = u[3]
        n_rpm = u[4:]

        # Compute propeller forces
        C_T2C = calculate_dcm(order=[2, 3], angles=[delta_s, delta_r])

        D_prop = 0.14
        n_rps = n_rpm / 60   
        Va = 0.944 * self.U

        KT_0 = 0.4566
        KQ_0 = 0.0700
        KT_max = 0.1798
        KQ_max = 0.0312
        Ja_max = 0.6632

        tau_prop = np.zeros(6)
        for i in range(len(n_rpm)):
            if n_rps[i] > 0:
                X_prop_i = self.rho*(D_prop**4)*(
                        KT_0*abs(n_rps[i])*n_rps[i] +
                        (KT_max-KT_0)/Ja_max * (Va/D_prop) * abs(n_rps[i]))
                K_prop_i = self.rho * (D_prop**5) * (
                        KQ_0 * abs(n_rps[i]) * n_rps[i] +
                        (KQ_max-KQ_0)/Ja_max * (Va/D_prop) * abs(n_rps[i]))
            else:
                X_prop_i = self.rho * (D_prop ** 4) * KT_0 * abs(n_rps[i]) * n_rps[i]
                K_prop_i = self.rho * (D_prop ** 5) * KQ_0 * abs(n_rps[i]) * n_rps[i]

            F_prop_b = C_T2C @ np.array([X_prop_i, 0, 0])
            r_prop_i = C_T2C @ self.propellers.r_t_p_sh[i] - self.p_OC_O
            M_prop_i = np.cross(r_prop_i, F_prop_b) \
                        + np.array([(-1)**i * K_prop_i, 0, 0])  # the -1 is because we have counter rotating
                                    # propellers that are supposed to cancel out the propeller induced
                                    # momentum
            tau_prop_i = np.concatenate([F_prop_b, M_prop_i])
            tau_prop += tau_prop_i

        return tau_prop


    def calculate_vbs_position(self, u):
        """
        Control input is scaled between 0 and 100. This converts it into the actual position
        s.t. we can calculate the amount of water in the VBS.
        u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        """
        x_vbs = (u[0]/100) * self.vbs.l_vbs_l
        return x_vbs


    def calculate_lcg_position(self, u):
        """
        Calculate the position of the LCG based on control input. The control
        input is scaled between 0 and 100. This function converts it to the
        actual physical location.
        """

        p_LcgPos_LcgO = np.array([(u[1]/100) * self.lcg.l_lcg_l, # Position of the LCG w.r.t fixed LCG point
                                 0, 0])
        p_OLcg_O = self.p_OLcgPos_O + p_LcgPos_LcgO

        return p_OLcg_O

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

        om = nu[3:6]  # Angular velocity
        gamma = 100

        ## From Fossen 2021, eq. 2.78:
        q0, q1, q2, q3 = q
        T_q_n_b = 0.5 * np.array([
                                 [-q1, -q2, -q3],
                                 [q0, -q3, q2],
                                 [q3, q0, -q1],
                                 [-q2, q1, q0]
                                 ])
        q_dot = T_q_n_b @ om + gamma/2 * (1 - q.T.dot(q)) * q

        return np.concatenate([pos_dot, q_dot])


    def actuator_dynamics(self, u_cur, u_ref):
        """
        Compute the actuator dynamics.
        delta_X and rpmX are assumed to be instantaneous

        u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]

        """

        u_dot = np.zeros(6)

        u_dot = (u_ref - u_cur)/self.dt

        if np.abs(u_dot[0]) > self.vbs.x_vbs_dot_max:
            u_dot[0] = self.vbs.x_vbs_dot_max * np.sign(u_dot[0])
        if np.abs(u_dot[1]) > self.lcg.x_lcg_dot_max:
            u_dot[1] = self.lcg.x_lcg_dot_max * np.sign(u_dot[1])

        return u_dot


