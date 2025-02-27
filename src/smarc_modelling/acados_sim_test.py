#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_closed_loop.py in getting started
# The NMPC base will exist in this script
#---------------------------------------------------------------------------------
import sys
import os
# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver


def setup(x0, Fmax, N_horizon, Tf, RTI=False):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    sam = SAM_casadi()

    # ------------------ MODEL EXTRACTION ---------------------
    model = sam.export_dynamics_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()


    # -------------------- Set costs ---------------------------
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q = np.eye(nx + nu)
    R = np.eye(nu)

    # Stage costs
    ocp.cost.W = ca.diagcat(Q, R).full() #scipy.linalg.block_diag
    ocp.cost.yref  = np.zeros((nx + nu,))
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.W_e = Q

    # Terminal cost
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref_e = np.zeros((nx,))


    # ---------------- Constraints ---------------------
    # set constraints
    # ocp.constraints.lbu = np.array([-Fmax])
    # ocp.constraints.ubu = np.array([+Fmax])

    ocp.constraints.x0 = x0
    #ocp.constraints.idxbu = np.array([0])

    # --------------- Solver options -------------------
    # set prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        ocp.solver_options.nlp_solver_max_iter = 150

    ocp.solver_options.qp_solver_cond_N = N_horizon


    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator


def main(use_RTI=False):

    x0 = np.zeros(25)
    x0[3] = 1
    Fmax = 80

    Tf = 1
    N_horizon = 20

    ocp_solver, integrator = setup(x0, Fmax, N_horizon, Tf, use_RTI)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    Nsim = 100
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0,:] = x0

    if use_RTI:
        t_preparation = np.zeros((Nsim))
        t_feedback = np.zeros((Nsim))

    else:
        t = np.zeros((Nsim))

    # do some initial iterations to start with a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = x0)

    # closed loop
    for i in range(Nsim):

        if use_RTI:
            # preparation phase
            ocp_solver.options_set('rti_phase', 1)
            status = ocp_solver.solve()
            t_preparation[i] = ocp_solver.get_stats('time_tot')

            # set initial state
            ocp_solver.set(0, "lbx", simX[i, :])
            ocp_solver.set(0, "ubx", simX[i, :])

            # feedback phase
            ocp_solver.options_set('rti_phase', 2)
            status = ocp_solver.solve()
            t_feedback[i] = ocp_solver.get_stats('time_tot')

            simU[i, :] = ocp_solver.get(0, "u")

        else:
            # solve ocp and get next control input
            simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])

            t[i] = ocp_solver.get_stats('time_tot')

        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings
    if use_RTI:
        # scale to milliseconds
        t_preparation *= 1000
        t_feedback *= 1000
        print(f'Computation time in preparation phase in ms: \
                min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
        print(f'Computation time in feedback phase in ms:    \
                min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
    else:
        # scale to milliseconds
        t *= 1000
        print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')
    print(np.shape(simX))
    # plot results
    model = ocp_solver.acados_ocp.model
    plot_pendulum(np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1), Fmax, simU, simX, latexify=False, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)

    ocp_solver = None


if __name__ == '__main__':
    main(use_RTI=False)

