import numpy as np
import matplotlib.pyplot as plt
from smarc_modelling.lib import *


def plot_function(x_axis, ref, simX, simU):
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
    plt.step(x_axis, simU[:, 0])
    plt.legend([r"VBS"])
    plt.ylabel("VBS derivative") 
    plt.grid()

    plt.subplot(4,3,5)
    plt.step(x_axis, simU[:, 2])
    plt.legend([r"Stern angle"])
    plt.ylabel(r" Degree derivative [$\degree/s$]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.step(x_axis, simU[:, 4])
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
    plt.step(x_axis, simU[:, 1])
    plt.legend([r"LCG"])
    plt.ylabel("LCG derivative") 
    plt.grid()

    plt.subplot(4,3,11)
    plt.step(x_axis, simU[:, 3])
    plt.legend([r"Rudder angle"])
    plt.ylabel(r" Degree derivative [$\degree/s$]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.step(x_axis, simU[:, 5])
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


    # Plot the trajectory
    ax.plot3D(x, y, z, label='Trajectory', lw=2, c='r')
    ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')
    #ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c='black')

    # Set axis limits
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 3])
    ax.set_box_aspect([1, 2, 1])  # Example: X:Y:Z ratio


    # Add directional arrows
    arrow_step = 15  # Adjust this value to control the spacing of the arrows
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



    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits and labels
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 3])
    ax.set_box_aspect([1, 2, 1])  # Example: X:Y:Z ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_zaxis()
    # Initialize the plot elements
    trajectory_line, = ax.plot([], [], [], label='Trajectory', lw=2, c='r')
    reference_line, = ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')

    # Update function for animation
    def update(frame):
        trajectory_line.set_data(simX[:frame, 0], simX[:frame, 1])
        trajectory_line.set_3d_properties(simX[:frame, 2])
        ax.view_init(elev=44.0, azim=-60.0)

        return trajectory_line,

    # Create the animation
    frames = len(simX)
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    # Save the animation as a GIF
    filename = "3D_trajectory.gif"
    ani.save(filename, writer='pillow')
    print(f"3D GIF saved as {filename}")


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

def part_plot_function(ref, simX, simU):
    x_axis = np.linspace(0, (0.1)*simX.shape[0], simX.shape[0])
    ref = ref[:simX.shape[0], :]
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
    plt.step(x_axis, simU[:, 0])
    plt.legend([r"VBS"])
    plt.ylabel("VBS derivative") 
    plt.grid()

    plt.subplot(4,3,5)
    plt.step(x_axis, simU[:, 2])
    plt.legend([r"Stern angle"])
    plt.ylabel(r" Degree derivative [$\degree/s$]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.step(x_axis, simU[:, 4])
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
    plt.step(x_axis, simU[:, 1])
    plt.legend([r"LCG"])
    plt.ylabel("LCG derivative") 
    plt.grid()

    plt.subplot(4,3,11)
    plt.step(x_axis, simU[:, 3])
    plt.legend([r"Rudder angle"])
    plt.ylabel(r" Degree derivative [$\degree/s$]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.step(x_axis, simU[:, 5])
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


    # Plot the trajectory
    ax.plot3D(x, y, z, label='Trajectory', lw=2, c='r')
    ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')
    #ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c='black')

    # Set axis limits
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 3])
    ax.set_box_aspect([1, 2, 1])  # Example: X:Y:Z ratio


    # Add directional arrows
    arrow_step = 15  # Adjust this value to control the spacing of the arrows
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



    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits and labels
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 3])
    ax.set_box_aspect([1, 2, 1])  # Example: X:Y:Z ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_zaxis()
    # Initialize the plot elements
    trajectory_line, = ax.plot([], [], [], label='Trajectory', lw=2, c='r')
    reference_line, = ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')

    # Update function for animation
    def update(frame):
        trajectory_line.set_data(simX[:frame, 0], simX[:frame, 1])
        trajectory_line.set_3d_properties(simX[:frame, 2])
        ax.view_init(elev=44.0, azim=-60.0)

        return trajectory_line,

    # Create the animation
    frames = len(simX)
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    # Save the animation as a GIF
    filename = "3D_trajectory.gif"
    ani.save(filename, writer='pillow')
    print(f"3D GIF saved as {filename}")


    # Show the plot
    plt.show()
