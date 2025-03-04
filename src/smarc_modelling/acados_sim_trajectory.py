#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_closed_loop.py in getting started
# The NMPC base will exist in this script
#---------------------------------------------------------------------------------
import sys
import os
import re 
# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

# Function to extract arrays from the RTF content
def extract_arrays_from_rtf(rtf_file_path):
    with open(rtf_file_path, "r") as file:
        rtf_content = file.read()

    # Regex pattern to match the arrays in the RTF content
    array_pattern = r'\[array\((.*?)\)\]'
    array_pattern = r'\ array\((.*?)\)'

    
    # Find all arrays in the RTF content
    arrays = re.findall(array_pattern, rtf_content, re.DOTALL)
    
    # Convert arrays to numpy arrays
    numpy_arrays = []
    for array_str in arrays:
        # Convert string to a numpy array
        # Remove unwanted characters like newline and extra spaces
        array_str = array_str.replace("\n", "").replace(" ", "")
        array_str = array_str.replace("\\", "").replace(" ", "")
        array_str = array_str.strip("[]")
        number_str_list = array_str.split(",")
        
        # Convert string to a list of floats
        array_list = np.array([float(i) for i in number_str_list])

        # Convert list to numpy array
        numpy_arrays.append(np.array(array_list))
    return numpy_arrays


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


def setup(x0, N_horizon, Tf, model, ocp):
    nx = model.x.rows()
    nu = model.u.rows()

    # -------------------- Set costs ---------------------------

    # State weight matrix
    Q_diag = np.ones(nx)
    Q_diag[ 0:3 ] = 2000
    Q_diag[ 3:7 ] = 1000
    Q_diag[ 7:10] = 500
    Q_diag[10:13] = 500
    Q_diag[13:15] = 1e-2
    Q_diag[15:17] = 1/50
    Q_diag[17:  ] = 1e-6
    Q = np.diag(Q_diag)

    # Control weight matrix - Costs set according to Bryson's rule (MPC course)
    R_diag = np.ones(nu)
    R_diag[ :2] = 1/(100**2)
    R_diag[2:4] = 1/(7**2)
    R_diag[4: ] = 1/(1000**2)
    R = np.diag(R_diag)

    # Stage costs
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.W = ca.diagcat(Q, R).full()
    # Set reference point - used only in the setup. The true references are declared in the sim. for-loop
    ref = np.zeros((nx + nu,))
    ocp.cost.yref  = ref
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    

    # Terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W_e = Q#np.zeros(np.shape(Q))
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref_e = ref[:nx]


    # ---------------- Constraints ---------------------
    ocp.constraints.x0 = x0
    ocp.constraints.lbu = np.array([0, 0, -7, -7, -1000, -1000])
    ocp.constraints.ubu = np.array([100, 100, 7, 7, 1000, 1000])
    ocp.constraints.idxbu = np.arange(nu)

    # --------------- Solver options -------------------
    # set prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = 80
    ocp.solver_options.tol    = 1e-6       # NLP tolerance
    ocp.solver_options.qp_tol = 1e-6       # QP tolerance
    #ocp.solver_options.nlp_solver_tol_eq = 1e-6

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator


def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Extract the CasADi model
    sam = SAM_casadi()
    model = sam.export_dynamics_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()


    # Declare the initial state
    # load trajectory
    rtf_file_path = "/home/admin/smarc_modelling/src/smarc_modelling/sam_example_trajectory.rtf"  # Replace with your actual file path
    # Extract numpy arrays from the RTF content
    trajectory = extract_arrays_from_rtf(rtf_file_path)
    x0 = trajectory[0]
 

    # Horizon parameters 
    Tf = 1
    N_horizon = 20
    update_factor = 20
    Nsim = np.size(trajectory, 0)*update_factor  # Simulation duration (no. of iterations)


    # Setup of the solver and integrator
    ocp_solver, integrator = setup(x0, N_horizon, Tf, ocp.model, ocp)

    simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control sequence
    simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated state
    simX[0,:] = x0

    # Array to store the time values
    t = np.zeros((Nsim))

    # Initialize the state and control vector as David does
    for stage in range(N_horizon + 1):
        ocp_solver.set(stage, "x", x0)
    for stage in range(N_horizon):
        ocp_solver.set(stage, "u", np.zeros(nu,))

    # do some initial iterations to start with a good initial guess - from the example
    # num_iter_initial = 5
    # for _ in range(num_iter_initial):
    #     ocp_solver.solve_for_x0(x0_bar = x0)

    # closed loop - simulation
    Uref = np.zeros(nu)
    for i in range(Nsim):
        # Update reference vector
        if i % update_factor == 0:
            ref = np.concatenate((trajectory[int(i/update_factor)], Uref))
            for stage in range(N_horizon):
                ocp_solver.set(stage, "yref", ref)
            ocp_solver.set(N_horizon, "yref", ref[:nx])
         

        # Set current state
        ocp_solver.set(0, "lbx", simX[i, :])
        ocp_solver.set(0, "ubx", simX[i, :])

        # solve ocp and get next control input
        status = ocp_solver.solve()
        ocp_solver.print_statistics()
        if status != 0:
            print(f" Note: acados_ocp_solver returned status: {status}")

        # simulate system
        t[i] = ocp_solver.get_stats('time_tot')
        simU[i, :]   = ocp_solver.get(0, "u")
        print(f"Nsim: {i}\n{np.round(simU[i, :],3)}")
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')


    # plot results
    x_axis = np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1)
    plot(x_axis, simX, simU)

    ocp_solver = None


if __name__ == '__main__':
    main()