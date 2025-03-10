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

def plot(x_axis, ref, simX, simU):
    ref_err = ref
    ref = ref[:13]  

    q = [ref[3], ref[4], ref[5], ref[6]]
    psi, theta, phi = gnc.quaternion_to_angles(q)

    reference = np.zeros(12)
    reference[ :3] = ref[ :3]
    reference[3:6] = np.array([phi, theta, psi])
    reference[6:]  = ref[7:]
    
    ref = reference
    for i in range(np.size(simX,0)-1):
        ref = np.vstack((ref, reference))

    n = len(simX)
    psi = np.zeros(n)
    theta = np.zeros(n)
    phi = np.zeros(n)

    for i in range(n):
        q = [simX[i, 3], simX[i, 4], simX[i, 5], simX[i, 6]]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)
    
    print(np.shape(simX))
    y_axis = simX[:, 0]
    for i in range(np.size(simX, 1)):
        if i in [3, 4, 5, 6]:
            if i == 3:
                y_axis = np.column_stack((y_axis, phi))
            elif i == 4:
                y_axis = np.column_stack((y_axis, theta))
            elif i == 5:
                y_axis = np.column_stack((y_axis, psi))
        else: 
            y_axis = np.column_stack((y_axis, simX[:,i]))
            

    plt.figure()
    plt.subplot(4,3,1)
    plt.plot(x_axis, simX[:, 0] )
    plt.plot(x_axis,  ref[:, 0], linestyle='--', color='r')
    plt.legend(["X", "X_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,2)
    plt.plot(x_axis, simX[:, 1] )
    plt.plot(x_axis,  ref[:, 1], linestyle='--', color='r')
    plt.legend(["Y", "Y_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,3)
    plt.plot(x_axis, simX[:, 2])
    plt.plot(x_axis,  ref[:, 2], linestyle='--', color='r')
    plt.legend(["Z", "Z_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,4)
    plt.plot(x_axis, simX[:, 7])
    plt.plot(x_axis,  ref[:, 6], linestyle='--', color='r')
    plt.legend([r"$\dotX$", r"$\dotX_{ref}$"])
    plt.ylabel("X Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,5)
    plt.plot(x_axis, simX[:, 8])
    plt.plot(x_axis,  ref[:, 7], linestyle='--', color='r')
    plt.legend([r"$\dotY$", r"$\dotY_{ref}$"])
    plt.ylabel("Y Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.plot(x_axis, simX[:, 9])
    plt.plot(x_axis,  ref[:, 8], linestyle='--', color='r')
    plt.legend([r"$\dotZ$", r"$\dotZ_{ref}$"])
    plt.ylabel("Z Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,7)
    plt.plot(x_axis, np.rad2deg(phi))
    plt.plot(x_axis, ref[:, 3], linestyle='--', color='r')
    plt.legend([r"$\phi$", r"$\phi_{ref}$"])
    plt.ylabel("Roll [deg]")    
    plt.grid()

    plt.subplot(4,3,8)
    plt.plot(x_axis, np.rad2deg(theta))
    plt.plot(x_axis, ref[:, 4], linestyle='--', color='r')
    plt.legend([r"$\theta$", r"$\theta_{ref}$"])
    plt.ylabel("Pitch [deg]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.plot(x_axis, np.rad2deg(psi))
    plt.plot(x_axis, ref[:, 5], linestyle='--', color='r')
    plt.legend([r"$\psi$", r"$\psi_{ref}$"])
    plt.ylabel("Yaw [deg]")
    plt.grid()

    plt.subplot(4,3,10)
    plt.plot(x_axis, simX[:, 10])
    plt.plot(x_axis,  ref[:, 9], linestyle='--', color='r')
    plt.legend([r"$\dot\phi$", r"$\dot\phi_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()

    plt.subplot(4,3,11)
    plt.plot(x_axis, simX[:, 11])
    plt.plot(x_axis,  ref[:, 10], linestyle='--', color='r')
    plt.legend([r"$\dot\theta$", r"$\dot\theta_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.plot(x_axis, simX[:, 12])
    plt.plot(x_axis,  ref[:, 11], linestyle='--', color='r')
    plt.legend([r"$\dot\psi$", r"$\dot\psi_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()
    plt.show()

    print(np.shape(y_axis))
    x_error = RMSE_calculation(y_axis, ref)

    # Error plots
    plt.figure()
    plt.subplot(4,3,1)
    plt.plot(x_axis, x_error[:, 0] )
    plt.legend(["X"])
    plt.ylabel("Position RMSE [m]")
    plt.grid()

    plt.subplot(4,3,2)
    plt.plot(x_axis, x_error[:, 1] )
    plt.legend(["Y"])
    plt.ylabel("Position RMSE [m]")
    plt.grid()

    plt.subplot(4,3,3)
    plt.plot(x_axis, x_error[:, 2])
    plt.legend(["Z"])
    plt.ylabel("Position RMSE [m]")
    plt.grid()

    plt.subplot(4,3,4)
    plt.plot(x_axis, x_error[:, 3])
    plt.legend([r"$\phi error$"])
    plt.ylabel("Roll [deg]")    
    plt.grid()

    plt.subplot(4,3,5)
    plt.plot(x_axis, x_error[:, 4])
    plt.legend([r"$\theta error$"])
    plt.ylabel("Pitch [deg]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.plot(x_axis, x_error[:, 5])
    plt.legend([r"$\psi error$"])
    plt.ylabel("Yaw [deg]")
    plt.grid()

    plt.subplot(4,3,7)
    plt.plot(x_axis[:-1], simU[:, 0])
    plt.legend([r"$VBS$"])
    plt.title("Control inputs")
    plt.ylabel("VBS input") 
    plt.grid()

    plt.subplot(4,3,8)
    plt.plot(x_axis[:-1], simU[:, 2])
    plt.legend([r"$Stern angle$"])
    plt.ylabel(r"Stern Input [$\degree$]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.plot(x_axis[:-1], simU[:, 4])
    plt.legend([r"RPM1$"])
    plt.ylabel("Motor RPM")
    plt.grid()

    plt.subplot(4,3,10)
    plt.plot(x_axis[:-1], simU[:, 1])
    plt.legend([r"$LCG$"])
    plt.ylabel("LCG input")
    plt.grid()

    plt.subplot(4,3,11)
    plt.plot(x_axis[:-1], simU[:, 3])
    plt.legend([r"$Rudder angle$"])
    plt.ylabel(r"Rudder Input [$\degree$]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.plot(x_axis[:-1], simU[:, 5])
    plt.legend([r"RPM2$"])
    plt.ylabel("Motor RPM")
    plt.grid()

    plt.show()

def RMSE_calculation(var, ref):
    cumulative_rmse = np.empty(np.shape(var))
    for i in range(np.size(ref,1)-1):      # Loop over all references
        sum_of_squared_errors = 0
        rmse_k = np.empty([])
        for k in range(np.size(var, 0)-1):  # Loop over all time steps
            # Calculate squared error for the current time step
            squared_error = (var[k, i] - ref[k, i])**2
            sum_of_squared_errors += squared_error
            
            # Calculate cumulative RMSE
            rmse_k = np.append(rmse_k, np.sqrt(sum_of_squared_errors / (k+1)))
        cumulative_rmse[:,i] = rmse_k
    return cumulative_rmse


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
    ocp.cost.W_e = Q#np.zeros(np.shape(Q))
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

    # Set constraints on the states
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

    # Declare the initial state
    x0 = np.zeros(nx)
    x0[0] = 0 
    x0[3] = 1       # Must be 1 (quaternions)
    x0[17:] = 1e-6
    x0[7]   = 1e-6
    # x0[13] = 50
    # x0[14] = 50



    # Declare the reference state - Static point in this tests
    ref = np.zeros((nx + nu,))
    ref[0] = 0.2
    ref[1] = 0.0
    ref[2] = 0.0
    ref[3] = 1


    # Horizon parameters 
    Ts = 0.1            # Sampling Time
    N_horizon = 10       # Prediction Horizon
    Tf = Ts*N_horizon
    Nsim = 1000          # Simulation duration (no. of iterations) - sim. length is Ts*Nsim


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


    # closed loop - simulation
    for i in range(Nsim):
        # Update reference vector
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
        simU[i, :] = ocp_solver.get(0, "u")
        X_eval = ocp_solver.get(0, "x")
        print(f"Nsim: {i}\nX_opt {X_eval[13:]}")
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms:\n min: {np.min(t):.3f}\nmax: {np.max(t):.3f}\navg: {np.average(t):.3f}\nstdev: {np.std(t)}\nmedian: {np.median(t):.3f}')


    # plot results
    x_axis = np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1)
    plot(x_axis, ref, simX, simU)

    ocp_solver = None


if __name__ == '__main__':
    main()