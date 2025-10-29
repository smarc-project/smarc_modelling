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


def main(trajectory, Q, N_hor, T_s, map_instance):       ##CHANGE: input trajectory
    # Extract the CasADi model
    sam = SAM_casadi()

    # create ocp object to formulate the OCP
    Ts = T_s            # Sampling time ##CHANGE: from 0.2
    N_horizon = N_hor     # Prediction horizon
    build = True
    nmpc = NMPC(sam, Ts, N_horizon, update_solver_settings=build)
    # nmpc = NMPC_trajectory(sam, Ts, N_horizon, Q)   ## CHANGE
    nx = nmpc.nx        # State vector length + control vector
    nu = nmpc.nu        # Control derivative vector length

    
    # Declare duration of sim. and the x_axis in the plots
    Nsim = (trajectory.shape[0])            # The sim length should be equal to the number of waypoints
    x_axis = np.linspace(0, Ts*Nsim, Nsim)

    simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control derivative
    simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated states


    # Declare the initial state
    x0 = trajectory[0] 
    simX[0,:] = x0

    # Augment the trajectory and control input reference 
    Uref = np.zeros((trajectory.shape[0], nu))  # Derivative reference - set to 0 to penalize fast control changes
    trajectory = np.concatenate((trajectory, Uref), axis=1) 

    # Run the MPC setup
    ocp_solver, integrator = nmpc.setup_path_planner(x0, map_instance)
    # ocp_solver, integrator = nmpc.setup(x0, map_instance)

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

        # Update reference vector
        # If the end of the trajectory has been reached, (ref.shape < N_horizon)
        # set the following waypoints in the horizon to the last waypoint of the trajectory
        for stage in range(N_horizon):
            #print(ref.shape[0], stage)
            if ref.shape[0] < N_horizon and ref.shape[0] != 0:
                ocp_solver.set(stage, "p", ref[ref.shape[0]-1,:])
            else:
                ocp_solver.set(stage, "p", ref[stage,:])

        # Set the terminal state reference
        ocp_solver.set(N_horizon, "yref", ref[-1,:nx])
 
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

