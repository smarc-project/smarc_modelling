#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_closed_loop.py in getting started
# The NMPC base will exist in this script
#---------------------------------------------------------------------------------
import sys
import csv
import os
# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from control import *

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi

def plot(x_axis, ref, simX, simU):
    Uref = ref[:, 13:]
    ref = ref[:,:13]  

    psi = np.zeros(np.size(ref, 0))
    theta = np.zeros(np.size(ref, 0))
    phi = np.zeros(np.size(ref, 0))
    for i in range(np.size(ref, 0)):
        q = [ref[i, 3], ref[i, 4], ref[i, 5], ref[i, 6]]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)

    reference = np.zeros((np.size(ref, 0), 12))
    reference[:, :3] = ref[:, :3]
    reference[:, 3] = phi
    reference[:, 4] = theta
    reference[:, 5] = psi
    reference[:, 6:]  = ref[:, 7:]
    
    ref = reference

    n = len(simX)
    psi = np.zeros(n)
    theta = np.zeros(n)
    phi = np.zeros(n)

    for i in range(n):
        q = [simX[i, 3], simX[i, 4], simX[i, 5], simX[i, 6]]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)
    
    y_axis = np.zeros(np.shape(simX))
    for i in range(np.size(simX, 1)):
        if i in [3, 4, 5, 6]:
            if i == 3:
                y_axis[:,i] = phi
            elif i == 4:
                y_axis[:,i] = theta
            elif i == 5:
                y_axis[:,i] = psi
        else: 
            y_axis[:,i] = simX[:,i]
            

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
    plt.plot(x_axis, np.rad2deg(ref[:, 3]), linestyle='--', color='r')
    plt.legend([r"$\phi$", r"$\phi_{ref}$"])
    plt.ylabel("Roll [deg]")    
    plt.grid()

    plt.subplot(4,3,8)
    plt.plot(x_axis, np.rad2deg(theta))
    plt.plot(x_axis, np.rad2deg(ref[:, 4]), linestyle='--', color='r')
    plt.legend([r"$\theta$", r"$\theta_{ref}$"])
    plt.ylabel("Pitch [deg]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.plot(x_axis, np.rad2deg(psi))
    plt.plot(x_axis, np.rad2deg(ref[:, 5]), linestyle='--', color='r')
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

    x_error = RMSE_calculation(y_axis, ref)

    # Error plots
    plt.figure()
    plt.subplot(2,3,1)
    plt.plot(x_axis, x_error[:, 0] )
    plt.legend(["X"])
    plt.ylabel("Position RMSE [m]")
    plt.grid()

    plt.subplot(2,3,2)
    plt.plot(x_axis, x_error[:, 1] )
    plt.legend(["Y"])
    plt.title("Errors")
    plt.ylabel("Position RMSE [m]")
    plt.grid()

    plt.subplot(2,3,3)
    plt.plot(x_axis, x_error[:, 2])
    plt.legend(["Z"])
    plt.ylabel("Position RMSE [m]")
    plt.grid()

    plt.subplot(2,3,4)
    plt.plot(x_axis, x_error[:, 3])
    plt.legend([r"$\phi$ error"])
    plt.ylabel("Roll [deg]")    
    plt.grid()

    plt.subplot(2,3,5)
    plt.plot(x_axis, x_error[:, 4])
    plt.legend([r"$\theta$ error"])
    plt.ylabel("Pitch [deg]")
    plt.grid()

    plt.subplot(2,3,6)
    plt.plot(x_axis, x_error[:, 5])
    plt.legend([r"$\psi$ error"])
    plt.ylabel("Yaw [deg]")
    plt.grid()

    # Control Inputs
    plt.figure()
    plt.subplot(4,3,1)
    plt.step(x_axis, simX[:, 13])
    plt.step(x_axis, Uref[:, 0], linestyle='--', color='r')
    plt.legend([r"VBS", r"VBS_{ref}"])
    plt.ylabel("VBS input") 
    plt.grid()

    plt.subplot(4,3,2)
    plt.step(x_axis, simX[:, 15])
    plt.step(x_axis, Uref[:, 2], linestyle='--', color='r')
    plt.title("Control inputs")
    plt.legend([r"Stern angle", r"s_{ref}"])
    plt.ylabel(r" Degrees [$\degree$]")
    plt.grid()

    plt.subplot(4,3,3)
    plt.step(x_axis, simX[:, 17])
    plt.step(x_axis, Uref[:, 4], linestyle='--', color='r')

    plt.legend([r"RPM1", r"RPM1_{ref}"])
    plt.ylabel("Motor RPM")
    plt.grid()

    plt.subplot(4,3,4)
    plt.step(x_axis[:-1], simU[:, 0])
    plt.legend([r"VBS"])
    plt.ylabel("VBS derivative") 
    plt.grid()

    plt.subplot(4,3,5)
    plt.step(x_axis[:-1], simU[:, 2])
    plt.legend([r"Stern angle"])
    plt.ylabel(r" Degree derivative [$\degree/s$]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.step(x_axis[:-1], simU[:, 4])
    plt.legend([r"RPM1"])
    plt.ylabel("RPM1 derivative")
    plt.grid()

    plt.subplot(4,3,7)
    plt.step(x_axis, simX[:, 14])
    plt.step(x_axis, Uref[:, 1], linestyle='--', color='r')

    plt.legend([r"LCG", r"LCG_{ref}"])
    plt.ylabel("LCG input")
    plt.grid()

    plt.subplot(4,3,8)
    plt.step(x_axis, simX[:, 16])
    plt.step(x_axis, Uref[:, 3], linestyle='--', color='r')

    plt.legend([r"Rudder angle", r"r_{ref}"])
    plt.ylabel(r" degrees [$\degree$]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.step(x_axis, simX[:, 18])
    plt.step(x_axis, Uref[:, 5], linestyle='--', color='r')

    plt.legend([r"RPM2", r"RPM2_{ref}"])
    plt.ylabel("Motor RPM")
    plt.grid()

    plt.subplot(4,3,10)
    plt.step(x_axis[:-1], simU[:, 1])
    plt.legend([r"LCG"])
    plt.ylabel("LCG derivative") 
    plt.grid()

    plt.subplot(4,3,11)
    plt.step(x_axis[:-1], simU[:, 3])
    plt.legend([r"Rudder angle"])
    plt.ylabel(r" Degree derivative [$\degree/s$]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.step(x_axis[:-1], simU[:, 5])
    plt.legend([r"RPM2"])
    plt.ylabel("RPM2 derivative")
    plt.grid()

    print(f"RMSE for each variable: {x_error[-1, :6]}")
    print(f"RMSE position norm: {np.linalg.norm(x_error[-1, :3])}")


    # State vectors
    # down-sampling the xyz data points
    x = simX[:,0]
    y = simX[:,1]
    z = simX[:,2]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits
    # ax.set_xlim([-21, 1])
    # ax.set_ylim([-11, 11])
    # ax.set_zlim([-1, 1])

    # Plot the trajectory
    ax.plot3D(x, y, z, label='Trajectory', lw=2, c='r')
    ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')
    #ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c='black')

    # Add directional arrows
    arrow_step = 10  # Adjust this value to control the spacing of the arrows
    for i in range(0, len(simX) - arrow_step, arrow_step):
        c = np.sqrt((simX[i + arrow_step, 0] - simX[i, 0])**2 + (simX[i + arrow_step, 1] - simX[i, 1])**2 + (simX[i + arrow_step, 2] - simX[i, 2])**2)
        ax.quiver(simX[i,0], simX[i, 1], simX[i, 2], 
                  np.cos(psi[i])*np.cos(theta[i]), 
                  np.sin(psi[i]), 
                  -np.sin(theta[i]), color='b', length=1, normalize=True)



    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Invert the z_axis
    ax.invert_zaxis()

    # Show the plot
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

def interpolate_trajectory(trajectory, update_factor):
    factors = np.linspace(0, 1, 3) # Create a reference between two current references. depends on upd_rate
    print(factors)
    for i in range(trajectory.shape[0]-1):
        if i != trajectory.shape[0]-1:
            for factor in factors[:-1]:
                if i == 0:
                    interpolated_vectors = trajectory[i] + factor * (trajectory[i+1] - trajectory[i])
                else:
                    interpolated_vectors = np.vstack([interpolated_vectors, trajectory[i] + factor * (trajectory[i+1] - trajectory[i])])
        else:
           for factor in factors:
                if i == 0:
                    interpolated_vectors = trajectory[i] + factor * (trajectory[i+1] - trajectory[i])
                else:
                    interpolated_vectors = np.vstack([interpolated_vectors, trajectory[i] + factor * (trajectory[i+1] - trajectory[i])])

    return interpolated_vectors
def main():
    # Extract the CasADi model
    sam = SAM_casadi(dt=Ts)
    model = sam.export_dynamics_model()
    nx = model.x.rows()
    nu = model.u.rows()
    #Nsim = 1200          # Simulation duration (no. of iterations) - sim. length is Ts*Nsim


    # create ocp object to formulate the OCP
    Ts = 0.2            # Sampling time
    N_horizon = 10      # Prediction horizon
    nmpc = NMPC_trajectory(model, Ts, N_horizon)

    # load trajectory 

    #file_path = "/home/admin/smarc_modelling/src/smarc_modelling/resolution01.csv"  # Replace with your actual file path
    file_path = "/home/admin/smarc_modelling/src/smarc_modelling/simonTrajectory.csv"
    trajectory = read_csv_to_array(file_path)
    update_factor = 1

    Nsim = (trajectory.shape[0]-1)
    x_axis = np.linspace(0, (Ts)*Nsim, Nsim+1)
    print(f"Trajectory shape: {trajectory.shape}")
    #trajectory = interpolate_trajectory(trajectory, update_factor)

    simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control sequence
    simX = np.zeros((Nsim+1, nx))   # Matrix to store the simulated state
    
    # Declare the initial state
    x0 = trajectory[0] 
    simX[0,:] = x0

    # Declare the reference state - Static point in this tests
    # Initialize ref
    ref = trajectory[1] #np.zeros((nx + nu,))
    Uref = np.zeros((trajectory.shape[0], nu))

    trajectory = np.concatenate((trajectory, Uref), axis=1)
    references = trajectory[0, :19] #ref
    ocp_solver, integrator = nmpc.setup(x0)
    # Initialize the state and control vector as David does
    for stage in range(N_horizon + 1):
        ocp_solver.set(stage, "x", x0)
    for stage in range(N_horizon):
        ocp_solver.set(stage, "u", np.zeros(nu,))

    # Array to store the time values
    t = np.zeros((Nsim))

    # closed loop - simulation
    Uref = np.zeros((nu,))

    for i in range(Nsim):
        print(f"Nsim: {i}")
        if i < (Nsim - N_horizon/update_factor):
            ref = trajectory[i:i+int(N_horizon/update_factor), :]
        else:
            ref = trajectory[i:, :]

        # Update reference vector
        for stage in range(N_horizon):
            if stage % update_factor == 0: 
                index = int(stage/update_factor)
                print(ref.shape[0], index)
                if ref.shape[0] < int(N_horizon/update_factor) and ref.shape[0] != 0:
                    ocp_solver.set(stage, "p", ref[ref.shape[0]-1,:])
                else:
                    ocp_solver.set(stage, "p", ref[index,:])


        references = np.vstack([references, ref[0,:nx]])
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
        X_eval = ocp_solver.get(0, "x")
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms:\n min: {np.min(t):.3f}\nmax: {np.max(t):.3f}\navg: {np.average(t):.3f}\nstdev: {np.std(t)}\nmedian: {np.median(t):.3f}')


    # plot results
    plot(x_axis, references, simX, simU)

    ocp_solver = None


if __name__ == '__main__':
    main()