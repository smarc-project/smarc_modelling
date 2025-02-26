# Script for the acados NMPC model
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi as ca

class NMPC:
    def solve(self, model):
        self.ocp = AcadosOcp()
        self.model = model  # Must be an acados ocp-model
        
        # Horizon parameters
        N = 20
        Tf = 1.0
        self.ocp.solver_options.N_horizon = N
        self.ocp.solver_options.tf = Tf

        # Declaration of cost matrices
        nx = self.model.x.rows()
        nu = self.model.u.rows()
        print(nx, nu)
        Q = np.eye(nx)
        R = np.eye(nu)
        
        # path cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u)
        self.ocp.cost.yref = np.zeros((nx+nu))
        self.ocp.cost.W = ca.diagcat(Q, R).full()

        # terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.yref_e = np.zeros((nx))
        self.ocp.model.cost_y_expr_e = self.model.x
        self.ocp.cost.W_e = Q

        # set constraints
        self.ocp.constraints.x0 = np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        # set options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        self.ocp.solver_options.integrator_type = 'IRK'
        # ocp.solver_options.print_level = 1
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

        ocp_solver = AcadosOcpSolver(self.ocp)

        status = ocp_solver.solve()
        ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            raise Exception(f'acados returned status {status}.')

        # Create two empty arrays to store the solution in
        x_opt_values = np.zeros((N+1, nx))
        u_opt_values = np.zeros((N, nu))
        # Extract the values from the solution
        for i in range(N):
            x_opt_values[i,:] = ocp_solver.get(i, "x")
            u_opt_values[i,:] = ocp_solver.get(i, "u")
        x_opt_values[N,:] = ocp_solver.get(N, "x")