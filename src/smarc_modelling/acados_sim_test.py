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

def plot(x_axis, simX, simU):
    plt.figure()
    plt.subplot(4,2,1)
    plt.plot(x_axis, simX[:,:3])
    plt.legend(["X", "Y", "Z"])
    plt.ylabel("Position [m]")
    plt.grid()

    n = len(simX)
    psi = np.zeros(n)
    theta = np.zeros(n)
    phi = np.zeros(n)

    for i in range(n):
        q = [simX[i, 3], simX[i, 4], simX[i, 5], simX[i, 6]]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)


    plt.subplot(4,2,2)
    plt.plot(x_axis, np.rad2deg(psi), x_axis, np.rad2deg(theta), x_axis, np.rad2deg(phi))
    plt.legend(["roll", "pitch", "yaw"])
    plt.ylabel("Angle [deg]")
    plt.grid()

    plt.subplot(4,2,3)
    plt.plot(x_axis, simX[:,7:10])
    plt.legend(["u", "v", "w"])
    plt.ylabel("Velocity [m/s]")
    plt.grid()

    plt.subplot(4,2,4)
    plt.plot(x_axis, simX[:,10:13])
    plt.legend(["Roll", "Pitch", "Yaw"])
    plt.ylabel("Angular velocity")
    plt.grid()

    plt.subplot(4,2,5)
    plt.step(x_axis, simX[:,13:17])
    plt.legend(["VBS", "LCG", "d_s", "d_r"])
    plt.ylabel("Control 1")
    plt.grid()

    plt.subplot(4,2,6)
    plt.step(x_axis, simX[:,17:19])
    plt.legend(["RPM1", "RPM2"])
    plt.ylabel("Control 2")
    plt.grid()

    plt.subplot(4,2,7)
    plt.step(x_axis[:-1], simU[:,:4])
    plt.legend(["VBS", "LCG", "d_s", "d_r"])
    plt.ylabel("Control ref")
    plt.xlabel("Time [s]")
    plt.grid()

    plt.subplot(4,2,8)
    plt.step(x_axis[:-1], simU[:,4:])
    plt.legend(["RPM1", "RPM2"])
    plt.ylabel("Control ref")
    plt.xlabel("Time [s]")
    plt.grid()
    plt.show()


def setup(x0, N_horizon, Tf):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()
    sam = SAM_casadi()

    # ------------------ MODEL EXTRACTION ---------------------
    model = sam.export_dynamics_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()

    # -------------------- Set costs ---------------------------
    # Set reference point - same both for intermediate shooting nodes and terminal state
    ref = np.zeros((nx + nu,))
    ref[0] = 0.1
    ref[3] = 1      # Must be one for the quaternions

    # Adjust the state weight matrix
    Q_diag = np.ones(nx)
    Q_diag[:3] = 1e4
    Q = np.diag(Q_diag)

    # Adjust the control weight matrix
    R_diag = np.ones(nu)
    R = np.diag(R_diag)

    # Stage costs
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.W = ca.diagcat(Q, R).full() #scipy.linalg.block_diag
    ocp.cost.yref  = ref

    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.W_e = Q

    # Terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref_e = ref[:nx]


    # ---------------- Constraints ---------------------
    ocp.constraints.x0 = x0


    # --------------- Solver options -------------------
    # set prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10

    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = 150

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator


def main():
    # Declare the initial state
    x0 = np.zeros(19)
    x0[3] = 1

    # Horizon parameters 
    Tf = 1
    N_horizon = 20
    Nsim = 100  # Simulation duration (no. of iterations)

    # Setup of the solver and integrator
    ocp_solver, integrator = setup(x0, N_horizon, Tf)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control sequence
    simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated state
    simX[0,:] = x0

    # Array to store the time values
    t = np.zeros((Nsim))

    # Initialize as David does
    for stage in range(N_horizon + 1):
        ocp_solver.set(stage, "x", x0)
    for stage in range(N_horizon):
        ocp_solver.set(stage, "u", np.zeros(nu,))

    # do some initial iterations to start with a good initial guess - Used from the example
    # num_iter_initial = 5
    # for _ in range(num_iter_initial):
    #     ocp_solver.solve_for_x0(x0_bar = x0)

    # closed loop - simulation
    for i in range(Nsim):
        # Set current state
        ocp_solver.set(0, "lbx", simX[i, :])
        ocp_solver.set(0, "ubx", simX[i, :])

        # solve ocp and get next control input
        status = ocp_solver.solve()
        if status != 0:
            print(f" Note: acados_ocp_solver returned status: {status}")

        # simulate system
        t[i] = ocp_solver.get_stats('time_tot')
        simU[i, :]   = ocp_solver.get(0, "u")
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')


    # plot results
    x_axis = np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1)
    plot(x_axis, simX, simU)

    ocp_solver = None


if __name__ == '__main__':
    main()