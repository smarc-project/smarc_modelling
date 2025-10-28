# Script for the acados NMPC model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import casadi as ca
import os
from typing import Optional

#The original NMPC class. Uses hard constraints.
class NMPC:
    def __init__(self, casadi_model, Ts, N_horizon, update_solver_settings):
        '''
        :param casadi_model: The casadi model to be used
        :param Ts: Sampling interval
        :param N_horizon: Control horizon
        :param update_solver_settings: If True, the solver will be updated with the new settings.
        '''
        self.ocp   = AcadosOcp()
        self.model = self.export_dynamics_model(casadi_model)
        self.ocp.model = self.model
        self.nx = self.model.x.rows()
        self.nu = self.model.u.rows()
        self.Ts = Ts
        self.Tf = Ts*N_horizon
        self.N_horizon = N_horizon
        self.update_solver = update_solver_settings
        
    # Function to create a Acados model from the casadi model
    def export_dynamics_model(self, casadi_model):
        # Create symbolic state and control variables
        x_sym     = ca.MX.sym('x', 19,1)
        u_sym = ca.MX.sym('u_sym', 6,1)

        # Create symbolic derivative
        x_dot_sym = ca.MX.sym('x_dot', 19, 1)
        
        # Set up acados model
        model = AcadosModel()
        model.name = 'SAM_equation_system'
        model.x = x_sym
        model.xdot = x_dot_sym
        model.u = u_sym

        # Declaration of explicit and implicit expressions
        x_dot  = casadi_model.dynamics(export=True)    # extract casadi.MX function
        f_expl = ca.vertcat(x_dot(x_sym[:13], x_sym[13:]), u_sym)
        f_impl = x_dot_sym - f_expl
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl

        return model
    
    def setup(self):    
        print("\033[92mNMPC setup is running\033[0m")    
        nx = self.model.x.rows()
        nu = self.model.u.rows()

        # --------------------------- Cost setup ---------------------------------
        # State weight matrix
        Q_diag = np.ones(nx)
        Q_diag[ 0:2 ] = 100     # Position:         standard 10
        Q_diag[ 2 ] = 500       # z-Position:         standard 10
        Q_diag[ 3:7 ] = 10       # Quaternion:       standard 10
        Q_diag[ 7:10] = 1       # linear velocity:  standard 1
        Q_diag[10:13] = 1       # Angular velocity: standard 1

        # Control weight matrix - Costs set according to Bryson's rule
        Q_diag[13] = 1e-5            # VBS:      Standard: 1e-4
        Q_diag[14] = 1e-5            # LCG:      Standard: 1e-4
        Q_diag[ 15  ] = 5e2             # stern_angle:   Standard: 100
        Q_diag[ 16  ] = 1e3             # rudder_angle:  Standard: 100
        Q_diag[17:  ] = 1e-6            # RPM1 And RPM2: Standard: 1e-6
        Q_diag[13:  ] = Q_diag[13:  ]   # Adjustment to all control weights
        Q = np.diag(Q_diag)

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        R_diag = np.ones(nu)
        R_diag[0] = 1e-1        # VBS
        R_diag[1] = 1e-1        # LCG
        R_diag[2:4] = 1e2
        R_diag[4: ] = 1e-5
        R = np.diag(R_diag)*1e-3

        # Stage costs
        self.model.p = ca.MX.sym('ref_param', nx+nu,1)
        self.ocp.parameter_values = np.zeros((nx+nu,))

        self.ocp.cost.yref  = np.zeros((nx+nu,))        # Init ref point. The true references are declared in the controller for-loop
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.W = ca.diagcat(Q, R).full()
        self.ocp.model.cost_y_expr = self.x_error(self.model.x, self.model.u, self.model.p, terminal=False)
        
        # Terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.W_e = Q #np.zeros(np.shape(Q))
        self.ocp.model.cost_y_expr_e = self.x_error(self.model.x, self.model.u, self.ocp.model.p, terminal=True)
        self.ocp.cost.yref_e = np.zeros((nx,))

        # --------------------- Constraint Setup --------------------------
        vbs_dot = 200   # Maximum rate of change for the VBS
        lcg_dot = 50    # Maximum rate of change for the LCG

        # Declare initial state
        self.ocp.constraints.x0 = np.zeros((nx,)) # Initial state is zero. This is set in the sim. for-loop

        # Set constraints on the control rate of change
        self.ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot])
        self.ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot])
        self.ocp.constraints.idxbu = np.arange(2)

        # --- position bounds (NED: z positive down) ---
        # Tank limits in meters
        x_min, x_max = 0.0, 8.0
        y_min, y_max = -1.5, 1.5
        z_min, z_max = -0.5, 3.0   

        pos_lbx = np.array([x_min, y_min, z_min])
        pos_ubx = np.array([x_max, y_max, z_max])

        # --- actuator state bounds for x[13:19] = [x_vbs, x_lcg, δs, δr, rpm1, rpm2] ---
        act_lbx = np.array([  0.0,   0.0, -np.deg2rad(7), -np.deg2rad(7),  -400.0,  -400.0])
        act_ubx = np.array([100.0, 100.0,  np.deg2rad(7),  np.deg2rad(7),   400.0,   400.0])

        ## Hard Constraints
        idxbx = np.r_[ [0,1,2], [13,14,15,16,17,18] ]     # 9 indices total
        lbx   = np.r_[ pos_lbx, act_lbx ]                 # length 9
        ubx   = np.r_[ pos_ubx, act_ubx ]                 # length 9

        self.ocp.constraints.idxbx = idxbx
        self.ocp.constraints.lbx   = lbx
        self.ocp.constraints.ubx   = ubx

        ## Soft Constraints
        idxsbx = np.array([0, 1, 2])    # Index of constraints we want to slacken

        # soften exactly those same state bounds:
        self.ocp.constraints.idxsbx = idxsbx

        # penalty weights (size must equal len(idxsbx))
        Z_weight = 1e4
        z_weight = 1e1
        n_sb = idxsbx.size
        self.ocp.cost.Zl = Z_weight*np.ones(n_sb)
        self.ocp.cost.Zu = Z_weight*np.ones(n_sb)
        self.ocp.cost.zl = z_weight*np.ones(n_sb)
        self.ocp.cost.zu = z_weight*np.ones(n_sb)

        # ----------------------- Solver Setup --------------------------
        # set prediction horizon
        self.ocp.solver_options.N_horizon = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hpipm_mode = 'ROBUST'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.sim_method_newton_iter = 2 #3 default

        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 1 #80
        self.ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
        self.ocp.solver_options.qp_tol = 1e-6       # QP tolerance

        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        #self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'
        self.ocp.solver_options.levenberg_marquardt = 1e-2
        #self.ocp.solver_options.regularize_method = 'PROJECT'

        # Define the folder path for the .json and c_generated code inside the home directory
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "acados_generated_code")
        self.ocp.code_export_directory = save_dir

        # Make sure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Setup the solver
        solver_json = os.path.join(save_dir, 'acados_ocp_' + self.model.name + '.json')

        acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json, generate=self.update_solver, build=self.update_solver)

        # Simulation object based on OCP model.
        sim = AcadosSim()
        sim.model = self.model
        sim.parameter_values = np.zeros(25)

        sim.solver_options.T = 0.1
        sim.solver_options.integrator_type = 'ERK'

        sim_json = os.path.join(save_dir, 'acados_sim_' + self.model.name + '.json')

        acados_integrator = AcadosSimSolver(sim, json_file = sim_json, generate=self.update_solver, build=self.update_solver)

        return acados_ocp_solver, acados_integrator
    
    def setup_path_planner(self, x0, map_instance, Q):
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
        # Q = self.Q  ##CHANGE

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
        # Q_e = np.zeros(nx)
        # Q_e[ :3] = 100
        # Q_e[3:7] = 100
        # Q_e[7:10]= 100
        # Q_e[10:13]= 0
        # Q_e[13:] = 0
        # Q_e = np.diag(Q_e)
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
        pointA = self.compute_trajectory_ends(self.model.x, forward=True)    ## CHANGE
        pointB = self.compute_trajectory_ends(self.model.x, forward=False)
        goal_constraints_pointA = ca.vertcat(
        pointA[0],
        pointA[1],
        pointA[2]
        )
        constraints_point_B = ca.vertcat(
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
        self.ocp.model.con_h_expr = ca.vertcat(goal_constraints_pointA, constraints_point_B)
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
        x_ubx[  :13] = 400

        # Set constraints on the control
        x_ubx[13:15] = 100 
        x_ubx[15:17] = np.deg2rad(7)
        x_ubx[17:  ] = 400

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
        #root_files_dir = '/home/parallels/Desktop/smarc_modelling-master/src/smarc_modelling/motion_planning/MotionPrimitives'
        #package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        package_root = os.path.abspath(this_file_dir)
        codegen_dir = os.path.join(package_root, 'optimization_double_mpc')
        ocp_dir = os.path.join(codegen_dir, 'acados_ocp_')
        os.makedirs(codegen_dir, exist_ok=True)
        self.ocp.code_export_directory = codegen_dir
        print(f"ext package acados dir: {codegen_dir}") 

        #acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json, generate=False, build=False)
        acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = ocp_dir + self.model.name + '.json', generate=True, build=True)

        # create an integrator with the same settings as used in the OCP solver.
        #acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json)
        acados_integrator = AcadosSimSolver(self.ocp, json_file = ocp_dir + self.model.name + '.json')


        return acados_ocp_solver, acados_integrator
    

    def compute_trajectory_ends(self, state, distance=0.655, forward: Optional[bool]= True):
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
        norm_q = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= norm_q
        q1 /= norm_q
        q2 /= norm_q
        q3 /= norm_q

        # Forward direction in body frame (longitudinal axis)
        forward_body = ca.vertcat(1, 0, 0)  # X-axis in body frame

        # Rotation matrix from quaternion
        R = ca.vertcat(
            ca.horzcat(1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)),
            ca.horzcat(2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)),
            ca.horzcat(2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2))
        )

        # Transform to world frame
        forward_world = R @ forward_body

        # Normalize forward vector
        forward_norm = np.sqrt(forward_world[0]**2 + forward_world[1]**2 + forward_world[2]**2)
        forward_world /= forward_norm

        # Compute new point
        if forward: 
            new_point = ca.vertcat(x, y, z) + distance * forward_world
        else:
            new_point = ca.vertcat(x, y, z) - distance * forward_world

        return new_point


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
        q_error = ca.if_else(q_w < 0, -q_error, q_error)  # Ensure the quaternion error is positive

        # NOTE: usually I'd have ref - state, the standard closed loop, i.e.
        # Astroem 2019. Since this error is squared, it should work, too,
        # Liniger 2014 uses it in their vanilla MPC formulation
        pos_error = x[:3] - ref[:3] 
        vel_error = x[7:13] - ref[7:13]
        u_error   = x[13:19] - ref[13:19]
        

        # If the error for terminal cost is calculated, don't include delta_u
        if terminal:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error)
        else:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error, u) #delta_u(u))
        return x_error
   