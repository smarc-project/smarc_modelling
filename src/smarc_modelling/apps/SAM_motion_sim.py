#---------------------------------------------------------------------------------
# INFO:
# Script to generate motion primitives and control SAM to follow them. 
# The script stores the control values and position of SAM for each trajectory 
#---------------------------------------------------------------------------------
import sys
import csv
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from smarc_modelling.lib import *
from smarc_modelling.apps.primitive_generator import *
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
from smarc_modelling.control.control import NMPC_trajectory


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

def read_csv_to_array(file_path: str):
    """
    Reads a CSV file and converts the elements to a NumPy array.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    np.array: A NumPy array containing the CSV data.
    """
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            data.append([float(element) for element in row])

    
    return np.array(data)

def get_state_and_control():
    # Extract the CasADi model
    sam = SAM_casadi()


    # create ocp object to formulate the OCP
    Ts = 0.2            # Sampling time
    N_horizon = 10      # Prediction horizon
    nmpc = NMPC_trajectory(sam, Ts, N_horizon)
    nx = nmpc.nx        # State vector length + control vector
    nu = nmpc.nu        # Control derivative vector length

    
    # load trajectory - Replace with your actual file path
    trajectories = generate_primitives()
    input("Trajectories generated, press enter to continue:")

    state_list = []
    control_list = []
    for i, trajectory in enumerate(trajectories):
        print(f"Trajectory: {(i+1)}/{len(trajectories)}")

        # Declare duration of sim. and the x_axis in the plots
        trajectory = trajectory.T               # Transpose the trajectory matrix to fit the MPC input
        Nsim = (trajectory.shape[0])            # The sim length should be equal to the number of waypoints
   
        simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control derivative
        simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated states


        # Declare the initial state
        x0 = trajectory[0] 
        simX[0,:] = x0

        # Augment the trajectory and control input reference 
        Uref = np.zeros((trajectory.shape[0], nu))  # Derivative reference - set to 0 to penalize fast control changes
        trajectory = np.concatenate((trajectory, Uref), axis=1) 

        # Run the MPC setup if first run
        if i == 0:
            ocp_solver, integrator = nmpc.setup(x0)

        # Initialize the state and control vector as David does
        for stage in range(N_horizon + 1):
            ocp_solver.set(stage, "x", x0)
        for stage in range(N_horizon):
            ocp_solver.set(stage, "u", np.zeros(nu,))

        # Array to store the time values
        t = np.zeros((Nsim))

        # closed loop - simulation
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
            t[i] = ocp_solver.get_stats('time_tot')
            simU[i, :] = ocp_solver.get(0, "u")
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

        state_list.append(simX[:, :13])
        control_list.append(simX[:, 13:])
        print(len(state_list))

    return state_list, control_list

if __name__ == '__main__':
    states, control = get_state_and_control()
    print(control)