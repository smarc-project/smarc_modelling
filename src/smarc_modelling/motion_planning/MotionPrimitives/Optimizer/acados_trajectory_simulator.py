#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_closed_loop.py in getting started
# The NMPC base will exist in this script
#---------------------------------------------------------------------------------
import sys
import csv
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
# from smarc_modelling.motion_planning.MotionPrimitives.Optimizer.control import *
from smarc_modelling.control.control import *

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi


def join_trees_optimization(trajectory,N_hor, T_s, build):       ##CHANGE: input trajectory
    # Extract the CasADi model
    sam = SAM_casadi(dt=0.1)

    # create ocp object to formulate the OCP
    Ts = T_s            # Sampling time ##CHANGE: from 0.2
    N_horizon = N_hor     # Prediction horizon
    # build = False
    nmpc = NMPC(sam, Ts, N_horizon, update_solver_settings=build)
    # nmpc = NMPC_trajectory(sam, Ts, N_horizon, Q)   ## CHANGE
    nx = nmpc.nx        # State vector length + control vector
    nu = nmpc.nu        # Control derivative vector length

    
    # Declare duration of sim. and the x_axis in the plots
    Nsim = (trajectory.shape[0])            # The sim length should be equal to the number of waypoints
    x_axis = np.linspace(0, Ts*Nsim, Nsim)

    simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control derivative
    simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated states


    # Declare the initial state — pad to nx if CSV has fewer columns
    x0 = np.zeros(nx)
    x0[:trajectory.shape[1]] = trajectory[0]
    simX[0, :] = x0

    # Augment trajectory: pad to N_PHYS_STATES, then add theta + control columns
    n_phys = nmpc.N_PHYS_STATES
    if trajectory.shape[1] < n_phys:
        trajectory = np.concatenate(
            (trajectory, np.zeros((trajectory.shape[0], n_phys - trajectory.shape[1]))),
            axis=1,
        )
    theta_col = np.zeros((trajectory.shape[0], 1))
    Uref = np.zeros((trajectory.shape[0], nu))
    trajectory = np.concatenate((trajectory, theta_col, Uref), axis=1)

    # Run the MPC setup
    # ocp_solver, integrator = nmpc.setup_path_planner(x0, map_instance)
    ocp_solver, integrator = nmpc.setup()

    # Initialize the state and control vector as David does
    for stage in range(N_horizon + 1):
        ocp_solver.set(stage, "x", x0)
    for stage in range(N_horizon):
        ocp_solver.set(stage, "u", np.zeros(nu,))

    # Array to store the time values
    t = np.zeros((Nsim))

    # closed loop - simulation
    print(f"Starting Planner MPC SIM Loop")
    for i in range(Nsim):
        #print(f"Nsim: {i}")

        # extract the sub-trajectory for the horizon
        if i <= (Nsim - N_horizon):
            ref = trajectory[i:i + N_horizon, :]
        else:
            ref = trajectory[i:, :]

        # Build 34-element parameter vector per stage
        goal_pos = trajectory[-1, :3]
        t_hat_default = np.array([1.0, 0.0, 0.0])
        stage_yref = np.zeros(nmpc.n_stage_cost)
        stage_yref[5] = 0.5

        for stage in range(N_horizon):
            row = ref[min(stage, ref.shape[0] - 1), :]
            p = np.r_[row, goal_pos, t_hat_default, 0.0]
            ocp_solver.set(stage, "p", p)
            ocp_solver.set(stage, "yref", stage_yref)

        terminal_row = ref[-1, :]
        p_terminal = np.r_[terminal_row, goal_pos, t_hat_default, 0.0]
        ocp_solver.set(N_horizon, "p", p_terminal)
        ocp_solver.set(N_horizon, "yref", np.zeros(nmpc.n_terminal_cost))
 
        # Set current state
        ocp_solver.set(0, "lbx", simX[i, :])
        ocp_solver.set(0, "ubx", simX[i, :])

        # solve ocp and get next control input
        status = ocp_solver.solve()
        #ocp_solver.print_statistics()


        # simulate system
        t[i] = ocp_solver.get_stats('time_tot')
        simU[i, :] = ocp_solver.get(0, "u")
        X_eval = ocp_solver.get(0, "x")
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    list_waypoints = simX.tolist()
    return list_waypoints, status

