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

# Plot the trajectory
def save_csv(trajectory, t):
    """
    Saves a NumPy array to a CSV file.

    Parameters:
    trajectory (np.array): The NumPy array to save.
    file_path (str): The path to the CSV file.
    """
    np.savetxt("/home/admin/smarc_modelling/src/Trajectories/REPORT/q2/predhorizon/trajectory_1Np.csv"
               , trajectory, delimiter=',', header="X,Y,Z,phi,theta,psi,vx,vy,vz,p,q,r,control", comments='')
    # np.savetxt("/home/admin/smarc_modelling/src/Trajectories/REPORT/q2/samplingtime/time_07s.csv"
    #            , t, delimiter=',', header="time (ms)", comments='')
    print("CSV file saved successfully.")


def rmse(true, pred):
    """
    Compute Root Mean Square Error.

    Parameters:
    true (array-like): Actual values.
    pred (array-like): Predicted values.

    Returns:
    float: RMSE value.
    """
    print(true.shape)
    norm = np.sqrt(np.mean((np.linalg.norm(true[:,:3] - pred[:,:3],axis=1)) ** 2))

    x_true = np.array(true[:, 0])
    x_pred = np.array(pred[:, 0])
    z_true = np.array(true[:, 2])
    z_pred = np.array(pred[:, 2])
    y_true = np.array(true[:, 1])
    y_pred = np.array(pred[:, 1])   
    
    xrmse = np.sqrt(np.mean((x_true - x_pred) ** 2))
    yrmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    zrmse = np.sqrt(np.mean((z_true - z_pred) ** 2))
    print(f"x: {xrmse}\ny: {yrmse}\nz: {zrmse}\nnorm: {norm}\n")

def main():
    # Extract the CasADi model
    sam = SAM_casadi()


    # create ocp object to formulate the OCP
    Ts = 0.1           # Sampling time
    N_horizon = 30 #12      # Prediction horizon
    build = True
    nmpc = NMPC(sam, Ts, N_horizon, update_solver_settings=build)
    nx = nmpc.nx        # State vector length + control vector
    nu = nmpc.nu        # Control derivative vector length
    nc = 1
    
    # load trajectory - Replace with your actual file path
    #file_path = "/home/admin/smarc_modelling/src/Trajectories/trajectoryComplexity3.csv"
    #file_path = "/home/admin/smarc_modelling/src/Trajectories/Complexity2Trajectory_0.csv"
    #file_path = "/home/admin/smarc_modelling/src/Trajectories/case_medium_original.csv"
    #file_path = "/home/admin/smarc_modelling/src/Trajectories/REPORT/medium/case_medium.csv"
    #file_path = "/home/admin/smarc_modelling/src/Trajectories/report_update/easy/trajectories/case_easy0.csv"
    #file_path = "/home/parallels/ros2_ws/src/smarc2/behaviours/sam/sam_diving_controller/sam_diving_controller/trajectoryComplexity3.csv"
    #file_path = "/home/parallels/ros2_ws/src/smarc2/behaviours/sam/sam_diving_controller/sam_diving_controller/trajectories/good_trajectory2.csv"
    file_path = "/home/orin/colcon_ws/src/smarc2/behaviours/sam/sam_diving_controller/sam_diving_controller/trajectories/straight_trajectory_1.csv"


    #file_path = "/home/admin/smarc_modelling/src/Trajectories/resolution01.csv"  
    trajectory = read_csv_to_array(file_path)
    # Declare duration of sim. and the x_axis in the plots
    Tsim = 50
    Nsim = int(Tsim/Ts) #(trajectory.shape[0])            # The sim length should be equal to the number of waypoints
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
    ocp_solver, integrator = nmpc.setup()

    # Initialize the state and control vector as David does
    for stage in range(N_horizon + 1):
        ocp_solver.set(stage, "x", x0)
    for stage in range(N_horizon):
        ocp_solver.set(stage, "u", np.zeros(nu,))

    # Array to store the time values
    t = np.zeros((Nsim))


    ref = np.zeros((N_horizon,trajectory.shape[1]))
    print(f"ref: {ref.shape}")

    traj_index = 0
    traj_len = trajectory.shape[0]

    print(f"traj len: {traj_len}")

    ref_0 = np.zeros((Nsim,trajectory.shape[1]))

    # closed loop - simulation
    for i in range(Nsim):
        # extract the sub-trajectory for the horizon
        # NOTE: This is only based on the euclidean distance between two points
        # and doesn't consider the orientation. Ideally, we would have the
        # proper pose difference in SO(3) for that.

        # Get current position
        x_current = simX[i,0]
        y_current = simX[i,1]
        z_current = simX[i,2]


        # Check if we have enough trajectory left for the prediction horizon
        if traj_index + N_horizon < traj_len:
            # Get distance to current waypoint
            d_current = np.sqrt((x_current - trajectory[traj_index,0])**2 + 
                                (y_current - trajectory[traj_index,1])**2 +
                                (z_current - trajectory[traj_index,2])**2)

            # Get distance to next waypoint
            d_next = np.sqrt((x_current - trajectory[traj_index+1,0])**2 + 
                            (y_current - trajectory[traj_index+1,1])**2 +
                            (z_current - trajectory[traj_index+1,2])**2)

            # Set trajectory index accordingly
            if d_next < d_current:
                traj_index += 1

        # Get subtrajectory starting at closest waypoint
        #ref = trajectory[traj_index:traj_index + N_horizon, :]
        #ref_0[i,:] = ref[0,:]

        #if i <= (Nsim - N_horizon):
        if i + N_horizon <= traj_len:
            ref = trajectory[i:i + N_horizon, :]
        #elif i >= traj_len:
        #    ref = trajectory[-1,:]
        else:
            ref = np.zeros((N_horizon,trajectory.shape[1]))
            ref[:,:] = trajectory[-1, :]
        #ref[:,:] = trajectory[-1, :]
        ref_0[i,:] = ref[0,:]

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
        if i % nc == 0 and i < Nsim - (nc-1):
            status = ocp_solver.solve()
            #ocp_solver.print_statistics()
            if status != 0:
                #break
                print(f" i: {i}, Note: acados_ocp_solver returned status: {status}")

            # simulate system
            t[i] = ocp_solver.get_stats('time_tot')
            for k in range(nc):
                simU[i+k, :] = ocp_solver.get(k, "u")
        
        noise_vector = np.zeros(19)
        #noise_vector[0:3] = np.array([(np.random.random()-0.5)/10,(np.random.random()-0.5)/10, (np.random.random()-0.5)/10])
        simX[i+1, :] = integrator.simulate(x=simX[i, :]+noise_vector, u=simU[i, :])
        np.set_printoptions(precision=3)
        #print(f"simX: {simX[i,:]}")
        #print(f"simU: {simU[i,:]}")
        #print(f"i: {i}, traj_index: {traj_index}")
     

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms:\n min: {np.min(t):.3f}\nmax: {np.max(t):.3f}\navg: {np.average(t):.3f}\nstdev: {np.std(t)}\nmedian: {np.median(t):.3f}')


    # plot results
    # print(f"x_axis: {x_axis.shape}")
    # print(f"refs: {trajectory.shape}")
    # print(f"simX: {simX.shape}")
    # print(f"simU: {simU.shape}")

    # Extract the optimal control sequence
    optimal_u = simX[:, 13:]
    #save_csv(simX,t)
    #rmse(simX[:-1], trajectory)

    zero_values = np.zeros((Nsim, nx))
    zero_values[:,3] = 1.
    #plot.plot_function(x_axis, trajectory, zero_values, simU)
    #plot.plot_function(x_axis, trajectory, simX[:-1], simU)
    #plot.plot_function(x_axis, ref_0, simX[:-1], simU)
    ocp_solver = None


if __name__ == '__main__':
    main()
