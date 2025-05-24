# Script for the acados NMPC model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
from smarc_modelling.motion_planning.MotionPrimitives.ObstacleChecker import compute_A_point_forward    ## CHANGE
import numpy as np
import casadi as ca
import os
from casadi import vertcat, horzcat, sqrt


class NMPC:
    def __init__(self, model, Ts, N_horizon):
        '''
        Input:
        model == Casadi model
        Ts == Sampling Time
        N_horizon == control horizon
        '''
        self.ocp   = AcadosOcp()
        self.model = model
        self.ocp.model = self.model
        self.Ts    = Ts
        self.Tf    = Ts*N_horizon
        self.N_horizon = N_horizon
        
    def setup(self, x0):
        nx = self.model.x.rows()
        nu = self.model.u.rows()

        # --------------------------- Cost setup ---------------------------------
        # State weight matrix
        Q_diag = np.ones(nx)
        Q_diag[ 0:3 ] = 0        # Position
        Q_diag[ 3:7 ] = 0       # Quaternion
        Q_diag[ 7:10] = 0          # linear velocity
        Q_diag[10:13] = 0         # Angular velocity

        # Control weight matrix - Costs set according to Bryson's rule (MPC course)
        Q_diag[13:15] = 0        # VBS, LCG
        Q_diag[15:17] = 0       # stern_angle, rudder_angle
        Q_diag[17:  ] = 0        # RPM1 And RPM2
        Q = np.diag(Q_diag)

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        R_diag = np.ones(nu)
        R_diag[ :2] = 1e-1
        R_diag[2:4] = 1
        R_diag[4: ] = 1e-5
        R = np.diag(R_diag)

        # Stage costs
        self.model.p = ca.MX.sym('ref_param', nx+nu,1)
        self.ocp.parameter_values = np.zeros((nx+nu,))

        self.ocp.cost.yref  = np.zeros((nx+nu,))        # Init ref point. The true references are declared in the sim. for-loop
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.W = ca.diagcat(Q, R).full()
        self.ocp.model.cost_y_expr = self.x_error(self.model.x, self.model.u, self.model.p, terminal=False) #ca.vertcat(self.model.x, self.model.u)
        
        # Terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.W_e = Q #np.zeros(np.shape(Q))
        self.ocp.model.cost_y_expr_e = self.x_error(self.model.x, self.model.u, self.ocp.model.p, terminal=True)
        self.ocp.cost.yref_e = np.zeros((nx,))

        # --------------------- Constraint Setup --------------------------
        vbs_dot = 10    # Maximum rate of change for the VBS
        lcg_dot = 15    # Maximum rate of change for the LCG
        ds_dot  = 7     # Maximum rate of change for stern angle
        dr_dot  = 7     # Maximum rate of change for rudder angle
        rpm_dot = 1000  # Maximum rate of change for rpm

        # Declare initial state
        self.ocp.constraints.x0 = x0

        # Set constraints on the control rate of change
        self.ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot, -ds_dot, -dr_dot, -rpm_dot, -rpm_dot])
        self.ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot,  ds_dot,  dr_dot,  rpm_dot,  rpm_dot])
        self.ocp.constraints.idxbu = np.arange(nu)

        # Set constraints on the states
        x_ubx = np.ones(nx)
        x_ubx[  :13] = 1000

        # Set constraints on the control
        x_ubx[13:15] = 100 
        x_ubx[15:17] = 7
        x_ubx[17:  ] = 1000

        x_lbx = -x_ubx
        x_lbx[13:15] = 0

        self.ocp.constraints.lbx = x_lbx
        self.ocp.constraints.ubx = x_ubx
        self.ocp.constraints.idxbx = np.arange(nx)

        # ----------------------- Solver Setup --------------------------
        # set prediction horizon
        self.ocp.solver_options.N_horizon = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hpipm_mode = 'ROBUST'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.sim_method_newton_iter = 3 #3 default

        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 80
        self.ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
        self.ocp.solver_options.qp_tol = 1e-6       # QP tolerance

        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'

        solver_json = 'acados_ocp_' + self.model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json)

        return acados_ocp_solver, acados_integrator
    

    def x_error(self, x, u, ref, terminal):
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
        q2 = q_conj/ca.norm_2(q2)
        
        # q_error = q1 @ q2^-1
        q_w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q_x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q_y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q_z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        q_error = ca.vertcat(q_w, q_x, q_y, q_z)

        pos_error = x[:3] - ref[:3]
        vel_error = x[7:13] - ref[7:13]
        u_error   = x[13:19] - ref[13:19]
        

        # If the error for terminal cost is calculated, don't include delta_u
        if terminal:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error)
        else:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error, u) #delta_u(u))
        return x_error

class NMPC_trajectory:
    def __init__(self, casadi_model, Ts, N_horizon, Q):
        '''
        Input:
        casadi_model == Casadi model
        Ts == Sampling Time
        N_horizon == control horizon
        '''
        self.ocp   = AcadosOcp()
        self.model = self.export_dynamics_model(casadi_model)
        self.ocp.model = self.model
        self.nx = self.model.x.rows()
        self.nu = self.model.u.rows()
        self.Ts    = Ts
        self.Tf    = Ts*N_horizon
        self.N_horizon = N_horizon
        self.Q = Q  ##CHANGE
    
    ##CHANGE
    def compute_A_point_forward_casadi(self, state, distance=0.655):
        """
        Compute the point forward along the vehicle's longitudinal axis using CasADi.
        """
        # Get current state elements
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]

        # Normalize quaternion
        norm_q = sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= norm_q
        q1 /= norm_q
        q2 /= norm_q
        q3 /= norm_q

        # Forward direction in body frame (longitudinal axis)
        forward_body = vertcat(1, 0, 0)  # X-axis in body frame

        # Rotation matrix from quaternion
        R = vertcat(
            horzcat(1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)),
            horzcat(2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)),
            horzcat(2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2))
        )

        # Transform to world frame
        forward_world = R @ forward_body

        # Normalize forward vector
        forward_norm = sqrt(forward_world[0]**2 + forward_world[1]**2 + forward_world[2]**2)
        forward_world /= forward_norm

        # Compute new point
        new_point = vertcat(x, y, z) + distance * forward_world

        return new_point
    ##CHANGE
    def compute_B_point_backward_casadi(self, state, distance=0.655):
        """
        Compute the point backward along the vehicle's longitudinal axis using CasADi.
        """
        # Get current state elements
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]

        # Normalize quaternion
        norm_q = sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= norm_q
        q1 /= norm_q
        q2 /= norm_q
        q3 /= norm_q

        # Forward direction in body frame (longitudinal axis)
        forward_body = vertcat(1, 0, 0)  # X-axis in body frame

        # Rotation matrix from quaternion
        R = vertcat(
            horzcat(1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)),
            horzcat(2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)),
            horzcat(2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2))
        )

        # Transform to world frame
        forward_world = R @ forward_body

        # Normalize forward vector
        forward_norm = sqrt(forward_world[0]**2 + forward_world[1]**2 + forward_world[2]**2)
        forward_world /= forward_norm

        # Compute new point (backward)
        new_point = vertcat(x, y, z) - distance * forward_world

        return new_point
    
    # Function to create a Acados model from the casadi model
    def export_dynamics_model(self, casadi_model):
        # Create symbolic state and control variables
        x_sym     = ca.MX.sym('x', 19,1)
        u_ref_sym = ca.MX.sym('u_ref', 6,1)

        # Create symbolic derivative
        x_dot_sym = ca.MX.sym('x_dot', 19, 1)
        
        # Set up acados model
        model = AcadosModel()
        model.name = 'SAM_equation_system'
        model.x    = x_sym
        model.xdot = x_dot_sym
        model.u    = u_ref_sym

        # Declaration of explicit and implicit expressions
        x_dot  = casadi_model.dynamics(export=True)    # extract casadi.MX function
        f_expl = ca.vertcat(x_dot(x_sym[:13], x_sym[13:]), u_ref_sym)
        f_impl = x_dot_sym - f_expl
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl

        return model
    
    def setup(self, x0, map_instance):
        nx = self.model.x.rows()
        nu = self.model.u.rows()

        # --------------------------- Cost setup ---------------------------------
        # State weight matrix
        '''
        Q_diag = np.ones(nx)
        Q_diag[ 0:3 ] = 15e1         # Position
        Q_diag[ 3:7 ] = 15e2         # Quaternion
        Q_diag[ 7:10] = 13e1        # linear velocity
        Q_diag[10:13] = 10e1         # Angular velocity

        # Control weight matrix - Costs set according to Bryson's rule (MPC course)
        Q_diag[13:15] = 0        # VBS, LCG
        Q_diag[15:17] = 0        # stern_angle, rudder_angle
        Q_diag[17:  ] = 0        # RPM1 And RPM2
        Q_diag[13:  ] = Q_diag[13:  ]
        Q = np.diag(Q_diag)
        '''
        Q = self.Q  ##CHANGE

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        R_diag = np.ones(nu)
        R_diag[ :2] = 1e-3
        R_diag[2:4] = 1e0
        R_diag[4: ] = 1e-5
        R = np.diag(R_diag)

        # Stage costs
        self.model.p = ca.MX.sym('ref_param', nx+nu,1)
        self.ocp.parameter_values = np.zeros((nx+nu,))

        self.ocp.cost.yref  = np.zeros((nx+nu,))        # Init ref point. The true references are declared in the sim. for-loop
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.W = ca.diagcat(Q, R).full()
        self.ocp.model.cost_y_expr = self.x_error(self.model.x, self.model.u, self.model.p, terminal=False) #ca.vertcat(self.model.x, self.model.u)
        
        # Terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        Q_e = np.zeros(nx)
        Q_e[ :3] = 100
        Q_e[3:7] = 100
        Q_e[7:10]= 100
        Q_e[10:13]= 0
        Q_e[13:] = 0
        Q_e = np.diag(Q_e)
        self.ocp.cost.W_e = Q #np.zeros(np.shape(Q))
        self.ocp.model.cost_y_expr_e = self.x_error(self.model.x, self.model.u, self.ocp.model.p, terminal=True)
        self.ocp.cost.yref_e = np.zeros((nx,))

        # --------------------- Constraint Setup --------------------------
        vbs_dot = 10    # Maximum rate of change for the VBS
        lcg_dot = 15    # Maximum rate of change for the LCG
        ds_dot  = 7     # Maximum rate of change for stern angle
        dr_dot  = 7     # Maximum rate of change for rudder angle
        rpm_dot = 1000  # Maximum rate of change for rpm

        # Declare initial state
        self.ocp.constraints.x0 = x0

        # Set constraints on the control rate of change
        self.ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot, -ds_dot, -dr_dot, -rpm_dot, -rpm_dot])
        self.ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot,  ds_dot,  dr_dot,  rpm_dot,  rpm_dot])
        self.ocp.constraints.idxbu = np.arange(nu)

        # Set constraint x in XFREE
        pointA = self.compute_A_point_forward_casadi(self.model.x)    ## CHANGE
        pointB = self.compute_B_point_backward_casadi(self.model.x)
        goal_constraints_pointA = vertcat(
        pointA[0],
        pointA[1],
        pointA[2]
        )
        constraints_point_B = vertcat(
            pointB[0],
            pointB[1],
            pointB[2]
        )
        bound = 0.1
        xMax = map_instance["x_max"] - bound
        yMax = map_instance["y_max"] - bound
        zMax = map_instance["z_max"] - bound
        xMin = map_instance["x_min"] + bound
        yMin = map_instance["y_min"] + bound
        zMin = map_instance["z_min"] + bound
        self.ocp.model.con_h_expr = vertcat(goal_constraints_pointA, constraints_point_B)
        self.ocp.constraints.lh = np.array([
            xMin, yMin, zMin, 
            xMin, yMin, zMin 
        ])
        self.ocp.constraints.uh = np.array([
            xMax, yMax, zMax, 
            xMax, yMax, zMax
        ])

        # Set constraints on the states
        x_ubx = np.ones(nx)
        x_ubx[  :13] = 1000

        # Set constraints on the control
        x_ubx[13:15] = 100 
        x_ubx[15:17] = 7
        x_ubx[17:  ] = 1000

        x_lbx = -x_ubx
        x_lbx[13:15] = 0

        self.ocp.constraints.lbx = x_lbx
        self.ocp.constraints.ubx = x_ubx
        self.ocp.constraints.idxbx = np.arange(nx)
        self.ocp.constraints.lbx_e = x_lbx
        self.ocp.constraints.ubx_e = x_ubx
        self.ocp.constraints.idxbx_e = np.arange(nx)

        # ----------------------- Solver Setup --------------------------
        # set prediction horizon
        self.ocp.solver_options.N_horizon = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hpipm_mode = 'ROBUST'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.sim_method_newton_iter = 3 #3 default

        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 80
        self.ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
        self.ocp.solver_options.qp_tol = 1e-6       # QP tolerance

        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'

        solver_json = 'acados_ocp_' + self.model.name + '.json'

        # Set directory for code generation
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        codegen_dir = os.path.join(package_root, 'optimization_double_mpc')
        os.makedirs(codegen_dir, exist_ok=True)
        self.ocp.code_export_directory = codegen_dir
        print(f"ext package acados dir: {codegen_dir}") 

        acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json, generate=False, build=False)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json)


        return acados_ocp_solver, acados_integrator
    

    def x_error(self, x, u, ref, terminal):
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
        q2 = q_conj/ca.norm_2(q2)
        
        # q_error = q1 @ q2^-1
        q_w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q_x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q_y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q_z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        q_error = ca.vertcat(q_w, q_x, q_y, q_z)

        pos_error = x[:3] - ref[:3] #+ np.array([(np.random.random()-0.5)/5,(np.random.random()-0.5)/5, (np.random.random()-0.5)/5])
        vel_error = x[7:13] - ref[7:13]
        u_error   = x[13:19] - ref[13:19]
        

        # If the error for terminal cost is calculated, don't include delta_u
        if terminal:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error)
        else:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error, u) #delta_u(u))
        return x_error
