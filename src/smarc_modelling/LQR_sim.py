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
from LQR import *

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_LQR import SAM_LQR
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi


def plot(x_axis, ref, simX, simU, simX2, simU2):
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
    psi2 = np.zeros(n)
    theta2 = np.zeros(n)
    phi2 = np.zeros(n)

    for i in range(n):
        q1 = simX[i, 3]
        q2 = simX[i, 4]
        q3 = simX[i, 5]
        q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
        q = [q0, q1, q2, q3]
        psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)

    for i in range(n):
        q = [simX2[i, 3], simX2[i, 4], simX2[i, 5], simX2[i, 6]]
        psi2[i], theta2[i], phi2[i] = gnc.quaternion_to_angles(q)
    
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
    plt.plot(x_axis, simX2[:, 0] )
    plt.plot(x_axis,  ref[:, 0], linestyle='--', color='r')
    plt.legend(["X", "X_org", "X_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,2)
    plt.plot(x_axis, simX[:, 1] )
    plt.plot(x_axis, simX2[:, 1] )
    plt.plot(x_axis,  ref[:, 1], linestyle='--', color='r')
    plt.legend(["Y", "y_org" "Y_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,3)
    plt.plot(x_axis, simX[:, 2])
    plt.plot(x_axis, simX2[:, 2] )
    plt.plot(x_axis,  ref[:, 2], linestyle='--', color='r')
    plt.legend(["Z","z_org", "Z_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,4)
    plt.plot(x_axis, simX[:, 7])
    plt.plot(x_axis, simX2[:, 7])
    plt.plot(x_axis,  ref[:, 6], linestyle='--', color='r')
    plt.legend([r"$\dotX$",r"$\dotX_{org}$", r"$\dotX_{ref}$"])
    plt.ylabel("X Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,5)
    plt.plot(x_axis, simX[:, 8])
    plt.plot(x_axis, simX2[:, 8])

    plt.plot(x_axis,  ref[:, 7], linestyle='--', color='r')
    plt.legend([r"$\dotY$", r"$\dotY_{org}$", r"$\dotY_{ref}$"])
    plt.ylabel("Y Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.plot(x_axis, simX[:, 9])
    plt.plot(x_axis, simX2[:, 9])

    plt.plot(x_axis,  ref[:, 8], linestyle='--', color='r')
    plt.legend([r"$\dotZ$", r"$\dotZ_{org}$", r"$\dotZ_{ref}$"])
    plt.ylabel("Z Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,7)
    plt.plot(x_axis, np.rad2deg(phi))
    plt.plot(x_axis, np.rad2deg(phi2))
    plt.plot(x_axis, np.rad2deg(ref[:, 3]), linestyle='--', color='r')
    plt.legend([r"$\phi$", r"$\phi_{org}$", r"$\phi_{ref}$"])
    plt.ylabel("Roll [deg]")    
    plt.grid()

    plt.subplot(4,3,8)
    plt.plot(x_axis, np.rad2deg(theta))
    plt.plot(x_axis, np.rad2deg(theta2))
    plt.plot(x_axis, np.rad2deg(ref[:, 4]), linestyle='--', color='r')
    plt.legend([r"$\theta$", r"$\theta_{org}$",r"$\theta_{ref}$"])
    plt.ylabel("Pitch [deg]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.plot(x_axis, np.rad2deg(psi))
    plt.plot(x_axis, np.rad2deg(psi2))
    plt.plot(x_axis, np.rad2deg(ref[:, 5]), linestyle='--', color='r')
    plt.legend([r"$\psi$", r"$\psi_{org}$",r"$\psi_{ref}$"])
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
    plt.plot(x_axis, simX[:, 11])
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

    # # Control Inputs
    plt.figure()
    # plt.subplot(4,3,1)
    # plt.step(x_axis, simX[:, 13])
    # plt.legend([r"VBS"])
    # plt.ylabel("VBS input") 
    # plt.grid()

    # plt.subplot(4,3,2)
    # plt.step(x_axis, simX[:, 15])
    # plt.title("Control inputs")
    # plt.legend([r"Stern angle"])
    # plt.ylabel(r" Degrees [$\degree$]")
    # plt.grid()

    # plt.subplot(4,3,3)
    # plt.step(x_axis, simX[:, 17])
    # plt.legend([r"RPM1"])
    # plt.ylabel("Motor RPM")
    # plt.grid()

    plt.subplot(4,3,4)
    plt.step(x_axis, simU[:, 0])
    plt.legend([r"VBS"])
    plt.ylabel("VBS") 
    plt.grid()

    plt.subplot(4,3,5)
    plt.step(x_axis, simU[:, 2])
    plt.legend([r"Stern angle"])
    plt.ylabel(r" Degree [$\degree$]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.step(x_axis, simU[:, 4])
    plt.legend([r"RPM1"])
    plt.ylabel("RPM1")
    plt.grid()

    # plt.subplot(4,3,7)
    # plt.step(x_axis, simX[:, 14])
    # plt.legend([r"LCG"])
    # plt.ylabel("LCG input")
    # plt.grid()

    # plt.subplot(4,3,8)
    # plt.step(x_axis, simX[:, 16])
    # plt.legend([r"Rudder angle"])
    # plt.ylabel(r" degrees [$\degree$]")
    # plt.grid()

    # plt.subplot(4,3,9)
    # plt.step(x_axis, simX[:, 18])
    # plt.legend([r"RPM2"])
    # plt.ylabel("Motor RPM")
    # plt.grid()

    plt.subplot(4,3,10)
    plt.step(x_axis, simU[:, 1])
    plt.legend([r"LCG"])
    plt.ylabel("LCG") 
    plt.grid()

    plt.subplot(4,3,11)
    plt.step(x_axis, simU[:, 3])
    plt.legend([r"Rudder angle"])
    plt.ylabel(r" Degree [$\degree/s$]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.step(x_axis, simU[:, 5])
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
    ax.plot3D(simX2[:,0], simX2[:,1], simX2[:,2], label='True model', lw=2, c='b')

    ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')


    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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

def main():
    # Extract the CasADi model
    sam = SAM_LQR()
    sam_org = SAM_casadi()
    original_function = sam_org.dynamics()              # The origial sam dynamics (verified with numpy model)
    dynamics_function = sam.dynamics(export=True)   # The LQR model to be used.
    nx   = 12
    nu   = 6
    Nsim = 100                          # Simulation duration (no. of iterations) - sim. length is Ts*Nsim
    simU = np.zeros((Nsim+1, nu))       # Matrix to store the optimal control sequence
    simX = np.zeros((Nsim+1, nx))       # Matrix to store the simulated state
    simU2 = np.zeros((Nsim+1, 6))       # Matrix to store the model's control sequence
    simX2 = np.zeros((Nsim+1, 19))      # Matrix to store the model's simulated state

    # create LQR object to to access methods
    Ts = 0.1
    lqr = LQR(dynamics_function, Ts)

    # Declare the initial state
    x0 = np.zeros(nx)
    x0[0] = 0.1  
    x0[7] = 1e-9
    simX[0,:] = x0

    # Declare initial state for the verified model
    x02 = np.zeros(19)
    x02[3] = 1
    x02[7] = 1e-9
    simX2[0,:] = x02

    # Declare control initial state
    u0 = np.zeros(nu)
    u0[:2] = 50
    u0[2:4] = 0
    u0[4:6] = 10
    simU[0,:] = u0
    simU2[0,:] = u0

    # Declare the reference state - Static point in this tests -NOT USED CURRENTLY
    x_ref = np.zeros(nx)
    references = x_ref

    u_ref = np.zeros(nu)
    u_ref[:2] = 50 


    # Extract the jacobians for the linear dynamics
    A, B = lqr.create_linearized_dynamics(x_ref, u_ref)

    # Initial linearization points
    x_lin = x0
    u_lin = u0  # np.zeros(nu)

    # Array to store the time values - NOT USED ATM
    t = np.zeros((Nsim))


    # closed loop - simulation
    x = x0
    x2 = x02
    u = u0
    u2 = u

    A_lin = A(x_lin, u_lin)
    B_lin = B(x_lin, u_lin)
    print(f"A_lin shape: {np.shape(A_lin)}\n{A_lin}")
    print(f"B_lin shape: {np.shape(B_lin)}\n{B_lin}")
    print(f"----------------------- SIMULATION STARTS---------------------------------")
    A_lin, B_lin = lqr.continuous_to_discrete(A_lin, B_lin, Ts)

    # SIMULATION LOOP
    for i in range(Nsim):
        print("-------------------------------------------------------------")
        print(f"Nsim: {i}")
        print(f"X_lin: {x_lin}, \nU_lin: {u_lin}")
        A_lin = A(x_lin, u_lin)
        B_lin = B(x_lin, u_lin)
        A_lin, B_lin = lqr.continuous_to_discrete(A_lin, B_lin, Ts)
        # ab = np.concatenate([A_lin @ B_lin, B_lin], axis=1)
        # print("rank: ", np.linalg.matrix_rank(ab))
        # print(f"Cntrollability matrix:\n {ab}")

        L = lqr.compute_lqr_gain(A_lin, B_lin)
        u  = -L @ (x)
        print(f"u: {u}")
        xdot = A_lin @ (x) + B_lin @ (u)
        x = np.array(x + xdot*Ts).flatten()

        x2 = np.array(x2 + original_function(x2, u2)*Ts).flatten()

        simX[i+1,:] = x
        simU[i+1,:] = np.array(u).flatten()
        simX2[i+1,:] = x2
        simU2[i+1,:] = np.array(u).flatten()
        if i % 1 == 0:
            x_lin = x
            u_lin = u
        references = np.vstack([references, x_ref])
 

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms:\n min: {np.min(t):.3f}\nmax: {np.max(t):.3f}\navg: {np.average(t):.3f}\nstdev: {np.std(t)}\nmedian: {np.median(t):.3f}')


    # plot results
    x_axis = np.linspace(0, (Ts)*Nsim, Nsim+1)
    plot(x_axis, references, simX, simU, simX2, simU2)

if __name__ == '__main__':
    main()