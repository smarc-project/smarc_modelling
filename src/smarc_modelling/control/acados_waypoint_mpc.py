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
from smarc_modelling.control.control import *

from smarc_modelling.vehicles import *
from smarc_modelling.lib import plot
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """
    Converts Euler angles (roll, pitch, yaw) to a quaternion (x, y, z, w).

    Args:
        roll: Rotation around the X-axis (in degreees)
        pitch: Rotation around the Y-axis (in degreees)
        yaw: Rotation around the Z-axis (in degreees)

    Returns:
        A tuple (q_x, q_y, q_z, q_w) representing the quaternion.
    """
    cr = np.cos(np.deg2rad(roll) / 2)
    sr = np.sin(np.deg2rad(roll) / 2)
    cp = np.cos(np.deg2rad(pitch) / 2)
    sp = np.sin(np.deg2rad(pitch) / 2)
    cy = np.cos(np.deg2rad(yaw) / 2)
    sy = np.sin(np.deg2rad(yaw) / 2)

    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return (q_w, q_x, q_y, q_z)


def main():
    # Extract the CasADi model
    sam = SAM_casadi()

    # create ocp object to formulate the OCP
    Ts = 0.1           # Sampling time
    N_horizon = 10      # Prediction horizon
    nmpc = NMPC(sam, Ts, N_horizon, update_solver_settings=True)
    nx = nmpc.nx        # State vector length + control vector
    nu = nmpc.nu        # Control derivative vector length
    
    Nsim = 10

    simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control derivative
    simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated states


    # Declare the initial state
    x0 = np.array([1.469e+00, -4.231e-02,  4.699e-03, 1.000e+00, 0.000e+00,0.000e+00, 0.000e+00,    # eta
                   0.000e+00, 0.000e+00, -0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00,              # nu
                   0.000e+00,  5.100e+01,  0.000e+00,  0.000e+00,  1.000e-06, 1.000e-06])           # control

    wp = np.array([6.613, -0.046, 0.656, 0.000, 0.000, 0.000, 1.000])
    ref = np.zeros((N_horizon, (nx+nu)))
    #ref[:7] = wp

    simX[0,:] = x0
    simU[:,5] = 10

    # Run the MPC setup
    ocp_solver, integrator = nmpc.setup()




    # Initialize the state and control vector as David does
    for stage in range(N_horizon + 1):
        ocp_solver.set(stage, "x", x0)
    for stage in range(N_horizon):
        ocp_solver.set(stage, "u", np.zeros(nu,))

    # Array to store the time values
    t = np.zeros((Nsim))

    # closed loop - simulation
    for i in range(Nsim):

        # Update reference vector
        # If the end of the trajectory has been reached, (ref.shape < N_horizon)
        # set the following waypoints in the horizon to the last waypoint of the trajectory
        for stage in range(N_horizon):
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
        if status != 0:
            print(f" Note: acados_ocp_solver returned status: {status}")

        # simulate system
        np.set_printoptions(precision=3)
        t[i] = ocp_solver.get_stats('time_tot')
        simU[i, :] = ocp_solver.get(0, "u")

        
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

        print(f"simX: {simX[i,:]}, simU: {simU[i,:]}")
     

    #plot.plot_function(x_axis, trajectory, simX[:-1], simU)
    ocp_solver = None


if __name__ == '__main__':
    main()
