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
import matplotlib.pyplot as plt
from LQR import *

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.lib.plot import refplot

def refplot(ref):
    x_axis = np.linspace(0, (0.1)*ref.shape[0], ref.shape[0])
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

    plt.figure()
    plt.subplot(4,3,1)
    plt.plot(x_axis,  ref[:, 0], linestyle='--', color='black')
    plt.legend(["X", "X_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,2)
    plt.plot(x_axis,  ref[:, 1], linestyle='--', color='black')
    plt.legend(["Y", "Y_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,3)
    plt.plot(x_axis,  ref[:, 2], linestyle='--', color='black')
    plt.legend(["Z", "Z_ref"])
    plt.ylabel("Position [m]")
    plt.grid()

    plt.subplot(4,3,4)
    plt.plot(x_axis,  ref[:, 6], linestyle='--', color='black')
    plt.legend([r"$\dotX$", r"$\dotX_{ref}$"])
    plt.ylabel("X Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,5)
    plt.plot(x_axis,  ref[:, 7], linestyle='--', color='black')
    plt.legend([r"$\dotY$", r"$\dotY_{ref}$"])
    plt.ylabel("Y Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,6)
    plt.plot(x_axis,  ref[:, 8], linestyle='--', color='black')
    plt.legend([r"$\dotZ$", r"$\dotZ_{ref}$"])
    plt.ylabel("Z Velocity [m/s]")
    plt.grid()

    plt.subplot(4,3,7)
    plt.plot(x_axis, np.rad2deg(ref[:, 3]), linestyle='--', color='black')
    plt.legend([r"$\phi$", r"$\phi_{ref}$"])
    plt.ylabel("Roll [deg]")    
    plt.grid()

    plt.subplot(4,3,8)
    plt.plot(x_axis, np.rad2deg(ref[:, 4]), linestyle='--', color='black')
    plt.legend([r"$\theta$", r"$\theta_{ref}$"])
    plt.ylabel("Pitch [deg]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.plot(x_axis, np.rad2deg(ref[:, 5]), linestyle='--', color='black')
    plt.legend([r"$\psi$", r"$\psi_{ref}$"])
    plt.ylabel("Yaw [deg]")
    plt.grid()

    plt.subplot(4,3,10)
    plt.plot(x_axis,  ref[:, 9], linestyle='--', color='black')
    plt.legend([r"$\dot\phi$", r"$\dot\phi_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()

    plt.subplot(4,3,11)
    plt.plot(x_axis,  ref[:, 10], linestyle='--', color='black')
    plt.legend([r"$\dot\theta$", r"$\dot\theta_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()

    plt.subplot(4,3,12)
    plt.plot(x_axis,  ref[:, 11], linestyle='--', color='black')
    plt.legend([r"$\dot\psi$", r"$\dot\psi_{ref}$"])
    plt.ylabel("Angular Velocity [rad/s]")
    plt.grid()


    # Control Inputs
    plt.figure()
    plt.subplot(4,3,1)
    plt.step(x_axis, Uref[:, 0], linestyle='--', color='black')
    plt.legend([r"VBS", r"VBS_{ref}"])
    plt.ylabel("VBS input") 
    plt.grid()

    plt.subplot(4,3,2)
    plt.step(x_axis, Uref[:, 2], linestyle='--', color='black')
    plt.title("Control inputs")
    plt.legend([r"Stern angle", r"s_{ref}"])
    plt.ylabel(r" Degrees [$\degree$]")
    plt.grid()

    plt.subplot(4,3,3)
    plt.step(x_axis, Uref[:, 4], linestyle='--', color='black')

    plt.legend([r"RPM1", r"RPM1_{ref}"])
    plt.ylabel("Motor RPM")
    plt.grid()


    plt.subplot(4,3,7)
    plt.step(x_axis, Uref[:, 1], linestyle='--', color='black')
    plt.legend([r"LCG", r"LCG_{ref}"])
    plt.ylabel("LCG input")
    plt.grid()

    plt.subplot(4,3,8)
    plt.step(x_axis, Uref[:, 3], linestyle='--', color='black')
    plt.legend([r"Rudder angle", r"r_{ref}"])
    plt.ylabel(r" degrees [$\degree$]")
    plt.grid()

    plt.subplot(4,3,9)
    plt.step(x_axis, Uref[:, 5], linestyle='--', color='black')
    plt.legend([r"RPM2", r"RPM2_{ref}"])
    plt.ylabel("Motor RPM")
    plt.grid()

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot3D(ref[:, 0], ref[:, 1], ref[:, 2], linestyle='--', label='Reference', lw=1, c='black')

    # Set axis limits
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 3])
    ax.set_box_aspect([1, 2, 1])  # Example: X:Y:Z ratio


    # Add directional arrows
    arrow_step = 35  # Adjust this value to control the spacing of the arrows
    for i in range(0, len(ref) - arrow_step, arrow_step):
        c = np.sqrt((ref[i + arrow_step, 0] - ref[i, 0])**2 + (ref[i + arrow_step, 1] - ref[i, 1])**2 + (ref[i + arrow_step, 2] - ref[i, 2])**2)
        ax.quiver(ref[i,0], ref[i, 1], ref[i, 2], 
                  np.cos(psi[i])*np.cos(theta[i]), 
                  np.sin(psi[i]), 
                  -np.sin(theta[i]), color='black', length=0.7, normalize=True)

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.view_init(elev=35.0, azim=35.0)

    # Invert the z_axis
    ax.invert_zaxis()


    # Show the plot
    plt.show()

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

def save_csv(trajectory):
    """
    Saves a NumPy array to a CSV file.

    Parameters:
    trajectory (np.array): The NumPy array to save.
    file_path (str): The path to the CSV file.
    """
    np.savetxt("/home/admin/smarc_modelling/src/Trajectories/REPORT/easy/case_easy9.csv", trajectory, delimiter=',', header="X,Y,Z,phi,theta,psi,vx,vy,vz,p,q,r", comments='')
def main():
    # Declare reference trajectory
    file_path = "/home/admin/smarc_modelling/src/Trajectories/REPORT/easy/case_easy.csv"
    trajectory = read_csv_to_array(file_path)

    # HARD
    # trajectory[0, 0] = trajectory[0, 0] + (np.random.random()-0.5)
    # trajectory[0, 1] = trajectory[0, 1] - (np.random.random()/2)
    # trajectory[0, 2] = trajectory[0, 2] + (np.random.random()-0.5)

    # Medium
    # trajectory[0, 0] = trajectory[0, 0] + (np.random.random()/2)
    # trajectory[0, 1] = trajectory[0, 1] + (np.random.random()-0.5)
    # trajectory[0, 2] = trajectory[0, 2] + (np.random.random()-0.5)

    # Easy
    trajectory[0, 0] = trajectory[0, 0] + (np.random.random()-0.5)
    trajectory[0, 1] = trajectory[0, 1] - (np.random.random()/2)
    trajectory[0, 2] = trajectory[0, 2] + (np.random.random()-0.5)

    save_csv(trajectory)
    #refplot(trajectory)

if __name__ == '__main__':
    main()