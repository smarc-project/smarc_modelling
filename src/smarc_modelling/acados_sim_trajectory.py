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

def plot(x_axis, simX, simU, ref):
    plt.figure()
    plt.subplot(4,2,1)
    plt.plot(x_axis, simX[:,:3])
    plt.plot(x_axis, ref[:,:3], linestyle='--')
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
    plt.plot(x_axis, np.rad2deg(phi), x_axis, np.rad2deg(theta), x_axis, np.rad2deg(psi))
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
    ocp.cost.yref  = ref        # Init ref point. The true references are declared in the sim. for-loop
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.W = ca.diagcat(Q, R).full()
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    

    # Terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W_e = np.zeros(np.shape(Q))
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref_e = ref[:nx]


    # --------------------- Constraint Setup --------------------------
    vbs_dot = 10    # Maximum rate of change for the VBS
    lcg_dot = 15    # Maximum rate of change for the LCG
    ds_dot  = 7     # Maximum rate of change for stern angle
    dr_dot  = 7     # Maximum rate of change for rudder angle
    rpm_dot = 1000  # Maximum rate of change for rpm

    # Declare initial state
    ocp.constraints.x0 = x0

    # Set constraints on the control rate of change
    ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot, -ds_dot, -dr_dot, -rpm_dot, -rpm_dot])
    ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot,  ds_dot,  dr_dot,  rpm_dot,  rpm_dot])
    ocp.constraints.idxbu = np.arange(nu)

    # Set constraints on the states (no constraints)
    x_ubx = np.ones(nx)
    x_ubx[  :13] = 1000

    # Set constraints on the control
    x_ubx[13:15] = 100 
    x_ubx[15:17] = 7
    x_ubx[17:  ] = 1000

    x_lbx = -x_ubx
    x_lbx[13:15] = 0

    ocp.constraints.lbx = x_lbx
    ocp.constraints.ubx = x_ubx
    ocp.constraints.idxbx = np.arange(nx)

    # ----------------------- Solver Setup --------------------------
    # set prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode = 'ROBUST'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 3 #3 default

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 80
    ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
    ocp.solver_options.qp_tol = 1e-6       # QP tolerance

    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'


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

    # load trajectory
    rtf_file_path = "/home/admin/smarc_modelling/src/smarc_modelling/sam_example_trajectory.rtf"  # Replace with your actual file path
    trajectory = extract_arrays_from_rtf(rtf_file_path)
    trajectory[-1][7:13] = 0
    print(trajectory)
    # Horizon parameters 
    Tf = 1
    N_horizon = 10
    update_factor = 6 # Update the reference every n:th iteration
    Nsim = np.size(trajectory, 0)*update_factor +400 # Simulation duration (no. of iterations)

    # Declare the initial state
    x0 = trajectory[0]
    u0 = np.zeros(nu)
    
    
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
        ocp_solver.set(stage, "u", u0)

    # closed loop - simulation
    Uref = np.zeros(nu)
    for i in range(Nsim):
        # Update reference vector at every {update_factor} interval. 
        if i % update_factor == 0 and int(i/update_factor) < np.size(trajectory, 0):
            ref = np.concatenate((trajectory[int(i/update_factor)], Uref))
            for stage in range(N_horizon):
                ocp_solver.set(stage, "yref", ref)
            ocp_solver.set(N_horizon, "yref", ref[:nx])
         

        # Set current state
        ocp_solver.set(0, "lbx", simX[i, :])
        ocp_solver.set(0, "ubx", simX[i, :])

        # solve ocp and get next control input
        status = ocp_solver.solve()
        #ocp_solver.print_statistics()
        if status != 0:
            print(f" Note: acados_ocp_solver returned status: {status}")

        # simulate system
        t[i] = ocp_solver.get_stats('time_tot')
        simU[i, :]   = ocp_solver.get(0, "u")
        print(f"Nsim: {i}")
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')


    # plot results
    x_axis = np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1)

    # Reference
    trajectory_ref = np.reshape(trajectory[0], (1,19))
    for i in range(len(trajectory) - 1):
        interpolated_segment = np.linspace(trajectory[i], trajectory[i + 1], update_factor, endpoint=False)
        trajectory_ref = np.concatenate((trajectory_ref, interpolated_segment))

    if np.size(trajectory_ref, 0) != len(x_axis):
        diff = len(x_axis) - np.size(trajectory_ref, 0)
        for i in range(diff):
            trajectory_ref = np.vstack((trajectory_ref, trajectory[-1]))
    plot(x_axis, simX, simU, trajectory_ref)

    ocp_solver = None


if __name__ == '__main__':
    main()