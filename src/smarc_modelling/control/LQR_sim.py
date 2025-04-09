#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_closed_loop.py in getting started
# The NMPC base will exist in this script
#---------------------------------------------------------------------------------
import sys
import os
import csv

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from LQR import *

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_LQR import SAM_LQR
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi


def plot(x_axis, ref, u_ref, simX, simNl, simU):
    ref = ref[:,:13]  

    psi = np.zeros(np.size(ref, 0))
    theta = np.zeros(np.size(ref, 0))
    phi = np.zeros(np.size(ref, 0))
    for i in range(np.size(ref, 0)):
        q1 = ref[i, 3]
        q2 = ref[i, 4]
        q3 = ref[i, 5]
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
        q = [q0, q1, q2, q3]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)

    reference = np.zeros((np.size(ref, 0), 12))
    reference[:, :3] = ref[:, :3]
    reference[:, 3] = phi
    reference[:, 4] = theta
    reference[:, 5] = psi
    reference[:, 6:]  = ref[:, 6:]
    
    ref = reference

    n = len(simX)
    psi = np.zeros(n)
    theta = np.zeros(n)
    phi = np.zeros(n)
    psiNl = np.zeros(n)
    thetaNl = np.zeros(n)
    phiNl = np.zeros(n)

    for i in range(n):
        q1 = simX[i, 3]
        q2 = simX[i, 4]
        q3 = simX[i, 5]
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
        q = [q0, q1, q2, q3]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)

    for i in range(n):
        q1 = simNl[i, 3]
        q2 = simNl[i, 4]
        q3 = simNl[i, 5]
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
        q = [q0, q1, q2, q3]
        psiNl[i], thetaNl[i], phiNl[i] = gnc.quaternion_to_angles(q)


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
    plt.plot(x_axis, simNl[:, 0] )
    plt.plot(x_axis[:-1],  ref[:, 0], linestyle='--', color='r')
    plt.legend(["X", r"$X_{NL}$", r"$X_{ref}$"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,2)
    plt.plot(x_axis, simX[:, 1] )
    plt.plot(x_axis, simNl[:, 1] )
    plt.plot(x_axis[:-1],  ref[:, 1], linestyle='--', color='r')
    plt.legend(["Y", r"$Y_{NL}$", r"$Y_{ref}$"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,3)
    plt.plot(x_axis, simX[:, 2])
    plt.plot(x_axis, simNl[:, 2])
    plt.plot(x_axis[:-1],  ref[:, 2], linestyle='--', color='r')
    plt.legend(["Z", r"$Z_{NL}$", r"$Z_{ref}$"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,4)
    plt.plot(x_axis, simX[:, 6])
    plt.plot(x_axis, simNl[:, 6] )
    plt.plot(x_axis[:-1],  ref[:, 6], linestyle='--', color='r')
    plt.legend([r"$\dotX$",r"$\dotX_{NL}$", r"$\dotX_{ref}$"])
    plt.ylabel("X Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,5)
    plt.plot(x_axis, simX[:, 7])
    plt.plot(x_axis, simNl[:, 7])
    plt.plot(x_axis[:-1],  ref[:, 7], linestyle='--', color='r')
    plt.legend([r"$\dotY$", r"$\dotY_{NL}$", r"$\dotY_{ref}$"])
    plt.ylabel("Y Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.plot(x_axis, simX[:, 8])
    plt.plot(x_axis, simNl[:, 8])
    plt.plot(x_axis[:-1],  ref[:, 8], linestyle='--', color='r')
    plt.legend([r"$\dotZ$", r"$\dotZ_{NL}$", r"$\dotZ_{ref}$"])
    plt.ylabel("Z Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,7)
    plt.plot(x_axis, np.rad2deg(phi))
    plt.plot(x_axis, np.rad2deg(phiNl))
    plt.plot(x_axis[:-1], np.rad2deg(ref[:, 3]), linestyle='--', color='r')
    plt.legend([r"$\phi$", r"$\phi_{NL}$", r"$\phi_{ref}$"])
    plt.ylabel("Roll [deg]")    
    plt.grid()

    plt.subplot(4,3,8)
    plt.plot(x_axis, np.rad2deg(theta))
    plt.plot(x_axis, np.rad2deg(thetaNl))
    plt.plot(x_axis[:-1], np.rad2deg(ref[:, 4]), linestyle='--', color='r')
    plt.legend([r"$\theta$", r"$\theta_{NL}$",r"$\theta_{ref}$"])
    plt.ylabel("Pitch [deg]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.plot(x_axis, np.rad2deg(psi))
    plt.plot(x_axis, np.rad2deg(psiNl))
    plt.plot(x_axis[:-1], np.rad2deg(ref[:, 5]), linestyle='--', color='r')
    plt.legend([r"$\psi$", r"$\psi_{NL}$",r"$\psi_{ref}$"])
    plt.ylabel("Yaw [deg]")
    plt.grid()

    plt.subplot(4,3,10)
    plt.plot(x_axis, simX[:, 9])
    plt.plot(x_axis, simNl[:, 9])
    plt.plot(x_axis[:-1],  ref[:, 9], linestyle='--', color='r')
    plt.legend([r"$\dot\phi$", r"$\dot\phi_{NL}$", r"$\dot\phi_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()

    plt.subplot(4,3,11)
    plt.plot(x_axis, simX[:, 10])
    plt.plot(x_axis, simNl[:, 10])
    plt.plot(x_axis[:-1],  ref[:, 10], linestyle='--', color='r')
    plt.legend([r"$\dot\theta$", r"$\dot\theta_{NL}$", r"$\dot\theta_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.plot(x_axis, simX[:, 11])
    plt.plot(x_axis, simNl[:, 11])
    plt.plot(x_axis[:-1],  ref[:, 11], linestyle='--', color='r')
    plt.legend([r"$\dot\psi$", r"$\dot\psi_{NL}$", r"$\dot\psi_{ref}$"])
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

    # # Control Inputs
    plt.figure()
    plt.subplot(4,3,4)
    plt.step(x_axis, simU[:, 0])
    plt.plot(x_axis[:-1],  u_ref[:, 0], linestyle='--', color='r')
    plt.legend([r"VBS"])
    plt.ylabel("VBS") 
    plt.grid()

    plt.subplot(4,3,5)
    plt.step(x_axis, simU[:, 2])
    plt.plot(x_axis[:-1],  u_ref[:, 2], linestyle='--', color='r')
    plt.legend([r"Stern angle"])
    plt.ylabel(r" Degree [$\degree$]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.step(x_axis, simU[:, 4])
    plt.plot(x_axis[:-1],  u_ref[:, 4], linestyle='--', color='r')
    plt.legend([r"RPM1"])
    plt.ylabel("RPM1")
    plt.grid()

    plt.subplot(4,3,10)
    plt.step(x_axis, simU[:, 1])
    plt.plot(x_axis[:-1],  u_ref[:, 1], linestyle='--', color='r')
    plt.legend([r"LCG"])
    plt.ylabel("LCG") 
    plt.grid()

    plt.subplot(4,3,11)
    plt.step(x_axis, simU[:, 3])
    plt.plot(x_axis[:-1],  u_ref[:, 3], linestyle='--', color='r')
    plt.legend([r"Rudder angle"])
    plt.ylabel(r" Degree [$\degree$]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.step(x_axis, simU[:, 5])
    plt.plot(x_axis[:-1],  u_ref[:, 5], linestyle='--', color='r')
    plt.legend([r"RPM2"])
    plt.ylabel("RPM2")
    plt.grid()

    print(f"RMSE for each variable: {x_error[-1, :6]}")


    # State vectors
    # down-sampling the xyz data points
    x = simX[:,0]
    y = simX[:,1]
    z = simX[:,2]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot3D(x, y, z, label='Linearized', lw=2, c='r')
    #ax.plot3D(simNl[:,0], simNl[:,1], simNl[:,2], label='Nonlinear model', lw=2, c='b')

    ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')


    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_zaxis()
    ax.legend()

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

def main():
    # Extract the CasADi model
    sam = SAM_LQR()
    sam_casadi = SAM_casadi(dt=0.1)
    casadi_dynamics = sam_casadi.dynamics(export=True)
    dynamics_function = sam.dynamics(export=True)   # The LQR model to be used.
    nx   = 12
    nu   = 6    

    # create LQR object to to access methods
    Ts = 0.1
    lqr = LQR(dynamics_function, Ts)


    # Declare reference trajectory
    file_path = "/home/admin/smarc_modelling/src/Trajectories/resolution01.csv"  # Replace with your actual file path
    #file_path = "/home/admin/smarc_modelling/src/Trajectories/simonTrajectory.csv"
    file_path = "/home/admin/smarc_modelling/src/Trajectories/straight_trajectory.csv"
    trajectory = read_csv_to_array(file_path)
    Nsim = trajectory.shape[0]

    simU = np.zeros((trajectory.shape[0], nu))       # Matrix to store the optimal control sequence
    simX = np.zeros((trajectory.shape[0], nx))       # Matrix to store the simulated state
    simNonlinear = np.zeros((trajectory.shape[0], nx))       # Matrix to store the simulated state

    x_ref = trajectory[:, 0:3]
    x_ref = np.concatenate((x_ref, trajectory[:, 4:13]), axis=1)
    u_ref = trajectory[:, 13:]
    

    # Declare the initial state
    x0 = x_ref[0,:]
    x0[6] = 1e-6
    simX[0,:] = x0
    simNonlinear[0,:] = x0
    x_ref = np.delete(x_ref, 0, axis=0)     # Remove the initial state from the reference
    x_ref = np.delete(x_ref, 0, axis=0)     # approx. the same state (Ververy low ang. velocity in pitch 4e-3)


    # Declare control initial state
    u0 = u_ref[0,:]
    u0[4:] = 10
    simU[0,:] = u0
    u_ref = np.delete(u_ref, 0, axis=0)


    # Array to store the time values - NOT USED ATM
    t = np.zeros((Nsim))

    # closed loop - simulation
    x = x0
    u = u0

    # Init the jacobians for the linear dynamics, input is shape of vectors
    lqr.create_linearized_dynamics(x_ref.shape[1], u_ref.shape[1])

    # Initial linearization points
    x_lin = x0
    u_lin = u0

    # SIMULATION LOOP
    print(f"----------------------- SIMULATION STARTS---------------------------------")
    for i in range(Nsim-1):
        print("-------------------------------------------------------------")
        print(f"Nsim: {i}")

        x2, u = lqr.solve(x, u,  x_lin, u_lin)

        q1, q2, q3 = x[3:6]
        q0 = np.sqrt(np.abs(1 - q1**2 - q2**2 - q3**2))
        q = np.array([q0, q1, q2, q3])
        q = q/np.linalg.norm(q) 

        x = np.concatenate((x[:3], q, x[6:]))
        xdot = np.array(casadi_dynamics(x, u)).flatten()
        xdot = np.delete(xdot, 3, axis=0)
        simNonlinear[i+1,:] = simNonlinear[i,:] + xdot*Ts
        simX[i+1,:] = x2
        simU[i+1,:] = u

        if i < x_ref.shape[0]:
            x_lin = x_ref[i,:]
            u_lin = u_ref[i,:]
        if i == 0:
            references = x0.reshape(1,12)
        elif i >= x_ref.shape[0]:
            references = np.vstack([references, x_ref[-1,:]]) 
        else:
            references = np.vstack([references, x_ref[i,:]])  

 
        print(q/np.linalg.norm(q) ) 
  
        x=x2

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms:\n min: {np.min(t):.3f}\nmax: {np.max(t):.3f}\navg: {np.average(t):.3f}\nstdev: {np.std(t)}\nmedian: {np.median(t):.3f}')


    # plot results
    x_axis = np.linspace(0, (Ts)*Nsim, Nsim)
    plot(x_axis, references, u_ref, simX, simNonlinear, simU)

if __name__ == '__main__':
    main()