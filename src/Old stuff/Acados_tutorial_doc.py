# Acados tutorial doc

# Importera nödvändiga bibliotek
from acados_template import AcadosOcp, AcadosOcpSolver

# (i mitt fall även)
import casadi as ca
import numpy  as np


# create an ocp object to formulate the OCP - AcadosOCP är en klass
ocp = AcadosOcp()

# set model - modellen ska deklareras som ett attribut till acadosOCP-objektet
ocp.model = model

# modellen måste vara av följande slag - se pendulum_model.py för referens.
model = AcadosModel()

# set prediction horizon and sampling time tf(?)
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf

# Kostnad, ocp.cost -- attributer finns nedan
# Stage cost
ocp.cost.cost_type = 'NONLINEAR_LS' #Cost type at intermediate shooting nodes (1 to N-1) – string in {EXTERNAL, LINEAR_LS, NONLINEAR_LS, CONVEX_OVER_NONLINEAR}
ocp.model.cost_y_expr = ca.vertcat(model.x, model.u) # CasADi expression for nonlinear least squares;
ocp.cost.yref = np.zeros((nx+nu,))                   # reference at intermediate shooting nodes (1 to N-1). Default: np.array([]).
ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()         # Weighting matrix 

# Terminal cost
ocp.cost.cost_type_e = 'NONLINEAR_LS'
ocp.cost.yref_e = np.zeros((nx,))
ocp.model.cost_y_expr_e = model.x
ocp.cost.W_e = Q_mat

# Constraints are set through the constraint attribute of the ocp object - np.array
ocp.constraints.lbu = np.array([-Fmax])   # Lower bound on u for 0 to n-1 - lbx exists for x 1-n-1 //_0 för startnod _e för slutnod N
ocp.constraints.ubu = np.array([+Fmax])   # upper bound on u for 0 to n-1 - ubx exssts for x 1-n-1
ocp.constraints.idxbu = np.array([0])     # Indices of bounds on u for 0 to n-1 idxbx exists too

# initial state
ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])


# set solver options
# set options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
# PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
# PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
ocp.solver_options.integrator_type = 'IRK'
# ocp.solver_options.print_level = 1
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

ocp_solver = AcadosOcpSolver(ocp)


# Solve optimeringsproblemet. 
status = ocp_solver.solve()
ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

# Extrahera värden från lösningen
simX[i,:] = ocp_solver.get(i, "x")
simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")
