#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_ocp.py in getting started
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
from acados_template import AcadosOcp, AcadosOcpSolver


def main():
    # create ocp object and sam object
    ocp = AcadosOcp()
    sam = SAM_casadi()

    # ------------------ MODEL EXTRACTION ---------------------
    model = sam.export_dynamics_model()
    ocp.model = model                   # set model

    nx = model.x.rows()
    nu = model.u.rows()
   

    # ------------------- HORIZON OPTION -------------------------
    N = 20
    Tf = 5.0
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf


    # ------------------ COST DECLARATION -----------------------
    # Adjust the state weight matrix
    Q_diag = np.ones(nx)
    Q = np.diag(Q_diag)

    # Adjust the control weight matrix
    R_diag = np.ones(nu)
    R = np.diag(R_diag)

    # Stage cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    #ocp.cost.yref = np.zeros((nx+nu,))
    ocp.cost.yref = np.array([0.001, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.W = ca.diagcat(Q, R).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    #ocp.cost.yref_e = np.zeros((nx,))
    ocp.cost.yref_e = np.array([0.001, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q
    
    # ---------------------- CONSTRAINTS -------------------------
    # ocp.constraints.lbu = np.array([-Fmax])
    # ocp.constraints.ubu = np.array([+Fmax])
    # ocp.constraints.idxbu = np.array([0])
    # Initial state
    ocp.constraints.x0 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    # --------------------- SOLVER OPTIONS ------------------------
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

    plt.figure()
    plt.subplot(4,2,1)
    plt.plot(range(len(simX)), simX[:,:3])
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
    plt.plot(range(len(simX)), np.rad2deg(psi), range(len(simX)), np.rad2deg(theta), range(len(simX)), np.rad2deg(phi))
    plt.legend(["roll", "pitch", "yaw"])
    plt.ylabel("Angle [deg]")
    plt.grid()

    plt.subplot(4,2,3)
    plt.plot(range(len(simX)), simX[:,7:10])
    plt.legend(["u", "v", "w"])
    plt.ylabel("Velocity [m/s]")
    plt.grid()

    plt.subplot(4,2,4)
    plt.plot(range(len(simX)), simX[:,10:13])
    plt.legend(["Roll", "Pitch", "Yaw"])
    plt.ylabel("Angular velocity")
    plt.grid()

    plt.subplot(4,2,5)
    plt.step(range(len(simX)), simX[:,13:17])
    plt.legend(["VBS", "LCG", "d_s", "d_r"])
    plt.ylabel("Control 1")
    plt.grid()

    plt.subplot(4,2,6)
    plt.step(range(len(simX)), simX[:,17:19])
    plt.legend(["RPM1", "RPM2"])
    plt.ylabel("Control 2")
    plt.grid()

    plt.subplot(4,2,7)
    plt.step(range(len(simU)), simU[:,:4])
    plt.legend(["VBS", "LCG", "d_s", "d_r"])
    plt.ylabel("Control ref")
    plt.grid()

    plt.subplot(4,2,8)
    plt.step(range(len(simU)), simU[:,4:])
    plt.legend(["RPM1", "RPM2"])
    plt.ylabel("Control ref")
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()
