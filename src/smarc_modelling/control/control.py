# Script for the acados NMPC model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
import numpy as np
import casadi as ca
import os


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
        self.Ts    = Ts
        self.Tf    = Ts*N_horizon
        self.N_horizon = N_horizon
        self.update_solver = update_solver_settings
        
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
    
    def setup(self):    
        print("\033[92mNMPC setup is running\033[0m")    
        nx = self.model.x.rows()
        nu = self.model.u.rows()

        # --------------------------- Cost setup ---------------------------------
        # State weight matrix
        Q_diag = np.ones(nx)
        Q_diag[ 0:3 ] = 100     # Position:         standard 10
        Q_diag[ 3:7 ] = 0       # Quaternion:       standard 10
        Q_diag[ 7:10] = 1       # linear velocity:  standard 1
        Q_diag[10:13] = 1       # Angular velocity: standard 1

        # Control weight matrix - Costs set according to Bryson's rule
        Q_diag[13:15] = 1e-4            # VBS, LCG:      Standard: 1e-4
        Q_diag[ 15  ] = 5e1             # stern_angle:   Standard: 100
        Q_diag[ 16  ] = 5e1             # rudder_angle:  Standard: 100
        Q_diag[17:  ] = 1e-5            # RPM1 And RPM2: Standard: 1e-6
        Q_diag[13:  ] = Q_diag[13:  ]   # Adjustment to all control weights
        Q = np.diag(Q_diag)

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        R_diag = np.ones(nu)
        R_diag[ :2] = 1e-3
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

        # Set constraints on the states and input magnitudes
        x_ubx = np.ones(nu) # NOTE: Only controller magnitutes are constrained

        # Set constraints on the control magnitudes
        x_ubx[0:2] = 100 
        x_ubx[2:4] = np.deg2rad(7)
        x_ubx[4: ] = 600

        x_lbx = -x_ubx
        x_lbx[0:2] = 0

        self.ocp.constraints.lbx = x_lbx
        self.ocp.constraints.ubx = x_ubx
        self.ocp.constraints.idxbx = np.arange(13, nx)

        # ----------------------- Solver Setup --------------------------
        # set prediction horizon
        self.ocp.solver_options.N_horizon = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hpipm_mode = 'ROBUST'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.sim_method_newton_iter = 2 #3 default

        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 1 #80
        self.ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
        self.ocp.solver_options.qp_tol = 1e-6       # QP tolerance

        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'


        # Define the folder path for the .json and c_generated code inside the home directory
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "acados_generated_code")
        self.ocp.code_export_directory = save_dir

        # Make sure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Setup the solver
        solver_json = os.path.join(save_dir, 'acados_ocp_' + self.model.name + '.json')

        if self.update_solver == False:
            acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json, generate=False, build=False)
            acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json, generate=False, build=False)

        else:
            acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json)
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
        q_error = ca.if_else(q_w < 0, -q_error, q_error)  # Ensure the quaternion error is positive

        pos_error = x[:3] - ref[:3] 
        vel_error = x[7:13] - ref[7:13]
        u_error   = x[13:19] - ref[13:19]
        

        # If the error for terminal cost is calculated, don't include delta_u
        if terminal:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error)
        else:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error, u) #delta_u(u))
        return x_error
    
"""# NMPC class that uses soft constraints
class NMPC:
    def __init__(self, casadi_model, Ts, N_horizon, update_solver_settings):
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
        self.Ts = Ts
        self.Tf = Ts*N_horizon
        self.N_horizon = N_horizon
        self.update_solver = update_solver_settings
        
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
    
    def setup(self):    
        print("\033[92mNMPC_trajectory setup is running\033[0m")    
        nx = self.model.x.rows()
        nu = self.model.u.rows()

        # --------------------------- Cost setup ---------------------------------
        # State weight matrix
        Q_diag = np.ones(nx)
        Q_diag[ 0:3 ] = 10      # Position: standard 10
        Q_diag[ 3:7 ] = 10       # Quaternion: standard 10
        Q_diag[ 7:10] = 1       # linear velocity: standard 1
        Q_diag[10:13] = 1       # Angular velocity: standard 1

        # Control weight matrix - Costs set according to Bryson's rule
        Q_diag[13:15] = 1e-4            # VBS, LCG
        Q_diag[15:17] = 100            # stern_angle, rudder_angle
        Q_diag[17:  ] = 1e-6            # RPM1 And RPM2
        Q_diag[13:  ] = Q_diag[13:  ]   # Adjustment to control weights
        Q = np.diag(Q_diag)

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        R_diag = np.ones(nu)
        R_diag[ :2] = 1e-3
        R_diag[2:4] = 1e0
        R_diag[4: ] = 1e-5
        R = np.diag(R_diag)*1e-4

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
        Q_e[ :3] = 1
        Q_e[3:7] = 1
        Q_e[7:10]= 1
        Q_e[10:13]= 1
        Q_e[13:] = 0
        Q_e = np.diag(Q_e)
        self.ocp.cost.W_e = Q
        self.ocp.model.cost_y_expr_e = self.x_error(self.model.x, self.model.u, self.ocp.model.p, terminal=True)
        self.ocp.cost.yref_e = np.zeros((nx,))

        # --------------------- Constraint Setup --------------------------
        vbs_dot = 10    # Maximum rate of change for the VBS
        lcg_dot = 15    # Maximum rate of change for the LCG
        ds_dot  = 7     # Maximum rate of change for stern angle
        dr_dot  = 7     # Maximum rate of change for rudder angle
        rpm_dot = 1000  # Maximum rate of change for rpm

        # Declare initial state
        self.ocp.constraints.x0 = np.zeros((nx,)) # Initial state is zero. This is set in the sim. for-loop

        # Set constraints on the control rate of change
        # self.ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot, -ds_dot, -dr_dot, -rpm_dot, -rpm_dot])
        # self.ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot,  ds_dot,  dr_dot,  rpm_dot,  rpm_dot])
        # self.ocp.constraints.idxbu = np.arange(nu)
        self.ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot])
        self.ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot])
        self.ocp.constraints.idxbu = np.arange(2)

        # Set constraints on the states and input magnitudes
        x_ubx = np.ones(nu) # If state constraints are desired change nu to nx and declare for each state what is desired

        # Set constraints on the control
        x_ubx[0:2] = 100 
        x_ubx[2:4] = np.deg2rad(7)
        x_ubx[4: ] = 1000

        x_lbx = -x_ubx
        x_lbx[0:2] = 0

        # Soft constraints - implemented as https://github.com/acados/acados/blob/main/examples/acados_python/tests/soft_constraint_test.py
        # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
        rr= 1
        if rr == 1:
            self.ocp.constraints.lbx = x_lbx
            self.ocp.constraints.ubx = x_ubx
            self.ocp.constraints.idxbx = np.arange(13, nx)
            self.ocp.constraints.idxsbx = np.arange(nu)
        else:
            self.ocp.model.con_h_expr = self.model.x[13:19]
            self.ocp.constraints.lh = x_lbx
            self.ocp.constraints.uh = x_ubx
            # indices of slacked constraints within h
            self.ocp.constraints.idxsh = np.arange(nu)
        self.ocp.cost.Zl = 50*np.ones((nu,))
        self.ocp.cost.Zu = 50*np.ones((nu,))
        self.ocp.cost.zl = 5*np.ones((nu,))
        self.ocp.cost.zu = 5*np.ones((nu,))


        # ----------------------- Solver Setup --------------------------
        # set prediction horizon
        self.ocp.solver_options.N_horizon = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hpipm_mode = 'ROBUST'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.sim_method_newton_iter = 2 #3 default

        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.nlp_solver_max_iter = 80
        self.ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
        self.ocp.solver_options.qp_tol = 1e-6       # QP tolerance

        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'

        solver_json = 'acados_ocp_' + self.model.name + '.json'
        if self.update_solver == False:
            acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json, generate=False, build=False)

            # create an integrator with the same settings as used in the OCP solver. generate=False, build=False
            acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json, generate=False, build=False)

        else:
            acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json)
            acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json)

        return acados_ocp_solver, acados_integrator
    

    def x_error(self, x, u, ref, terminal):
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

        pos_error = x[:3] - ref[:3] #+ np.array([(np.random.random()-0.5)/5,(np.random.random()-0.5)/5, (np.random.random()-0.5)/5])
        vel_error = x[7:13] - ref[7:13]
        u_error   = x[13:19] - ref[13:19]
        

        # If the error for terminal cost is calculated, don't include delta_u
        if terminal:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error)
        else:
            x_error = ca.vertcat(pos_error, q_error, vel_error, u_error, u) #delta_u(u))
        return x_error
"""