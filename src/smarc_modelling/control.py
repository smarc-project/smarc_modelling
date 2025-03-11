# Script for the acados NMPC model
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import casadi as ca

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
        Q_diag[ 0:3 ] = 4e3
        Q_diag[ 3:7 ] = 4e3
        Q_diag[ 7:10] = 500
        Q_diag[10:13] = 500

        # Control weight matrix - Costs set according to Bryson's rule (MPC course)
        Q_diag[13:15] = 1e-6
        Q_diag[15:17] = 1/50
        Q_diag[17:  ] = 1e-6
        Q = np.diag(Q_diag)

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        R_diag = np.ones(nu)
        R_diag[ :2] = 4e-2
        R_diag[2:4] = 1
        R_diag[4: ] = 1e-5
        R = np.diag(R_diag)

        # Stage costs
        ref = np.zeros((nx + nu,))
        self.ocp.cost.yref  = ref        # Init ref point. The true references are declared in the sim. for-loop
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.W = ca.diagcat(Q, R).full()
        self.ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u)
        
        # Terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.W_e = Q#np.zeros(np.shape(Q))
        self.ocp.model.cost_y_expr_e = self.model.x
        self.ocp.cost.yref_e = ref[:nx]

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