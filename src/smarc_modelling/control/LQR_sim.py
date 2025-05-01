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
import time

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_LQR import SAM_LQR
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
from smarc_modelling.lib.plot import plot_function


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


def runge_kutta_4(f, x, u, dt):
    """
    Runge-Kutta 4th Order Integrator for solving ODEs.

    Parameters:
    f  : function - The dynamics function f(x, u) that returns dx/dt.
    x  : np.array - The current state vector.
    u  : np.array - The control input vector.
    dt : float    - The time step for integration.

    Returns:
    np.array - The next state vector after time step dt.
    """
    # Compute the four Runge-Kutta terms
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)

    # Combine the terms to compute the next state
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return np.array(x_next).flatten()

def main():
    # Extract the CasADi model
    sam = SAM_LQR()
    sam_casadi = SAM_casadi()
    casadi_dynamics = sam_casadi.dynamics(export=True)
    dynamics_function = sam.dynamics(export=True)   # The LQR model to be used (removed scalar part of the quaternion)
    nx   = 13
    nu   = 6    

    # create LQR object to to access methods
    Ts = 0.1
    lqr = LQR_integrator(dynamics_function, Ts)


    # Declare reference trajectory
    file_path = "/home/admin/smarc_modelling/src/Trajectories/REPORT/case_medium.csv"
    trajectory = read_csv_to_array(file_path)

    Nsim = trajectory.shape[0]
    simU = np.zeros((trajectory.shape[0]+1, nu))            # Matrix to store the optimal control sequence
    simNonlinear = np.zeros((trajectory.shape[0]+1, nx))    # Matrix to store the simulated state

    # Split the control and state reference and remove the scalar quaternion
    x_ref, u_ref = np.hsplit(trajectory, [13])
    
    # Declare the initial state and initial control
    x = x_ref[0,:]
    u = u_ref[0,:]
    simNonlinear[0,:] = x
    simU[0,:] = u

    # Initial linearization points
    x_lin = x_ref[1,:]
    u_lin = u_ref[1,:]

    # Init the jacobians for the linear dynamics, input is shape of vectors
    lqr.create_linearized_dynamics(x_ref.shape[1]-1, u_ref.shape[1])

    # Array to store the time values
    t = np.zeros((Nsim))

    # SIMULATION LOOP
    print(f"----------------------- SIMULATION STARTS---------------------------------")
    for i in range(Nsim):
        print("-------------------------------------------------------------")
        print(f"Nsim: {i}")
        time_start = time.time()
        u = lqr.solve(x, u, x_lin, u_lin)
        time_end = time.time()
        t[i] = time_end - time_start
        
        x_next = runge_kutta_4(casadi_dynamics, x, u, Ts)
        print(x_next[3:7])
        x_next[3:7] = x_next[3:7]/scipy.linalg.norm(x_next[3:7])  # Normalize quaternion
        
        if i < x_ref.shape[0]-1:
            x_lin = x_ref[i+1,:]
            u_lin = u_ref[i+1,:]

        x=x_next
        simNonlinear[i+1,:] = x_next
        simU[i+1,:] = u

    # evaluate timings
    t *= 1000  # scale to milliseconds
    print(f'Computation time in ms:\nmin: {np.min(t):.3f}\nmax: {np.max(t):.3f}\navg: {np.average(t):.3f}\nstdev: {np.std(t)}\nmedian: {np.median(t):.3f}')

    # plot results
    x_axis = np.linspace(0, (Ts)*Nsim, Nsim)
    sim = np.hstack([simNonlinear, simU])
    u_dot = np.zeros(simU.shape)
    #plot(x_axis, references, u_ref, simNonlinear[:-1], simU[:-1])
    plot_function(x_axis, trajectory, sim[:-1], u_dot[:-1])
if __name__ == '__main__':
    main()