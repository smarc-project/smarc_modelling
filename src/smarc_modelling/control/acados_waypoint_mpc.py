#---------------------------------------------------------------------------------
# INFO:
# Script to test the acados framework before putting it into the other scripts.
# It is based on the acados example minimal_example_closed_loop.py in getting started
# The NMPC base will exist in this script
#---------------------------------------------------------------------------------
import sys
import csv
import os
from tqdm import tqdm

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

from smarc_modelling.control.control import *

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi


class Sol():
    """
    Solver data class to match with Omid's plotting functions
    """
    def __init__(self, t, data) -> None:
        self.t = t
        self.y = data

def run_simulation():
    # Extract the CasADi model
    sam = SAM_casadi()

    # create ocp object to formulate the OCP
    Ts = 0.1           # Sampling time
    N_horizon = 10      # Prediction horizon
    nmpc = NMPC(sam, Ts, N_horizon, update_solver_settings=False)
    nx = nmpc.nx        # State vector length + control vector
    nu = nmpc.nu        # Control derivative vector length
    
    # Simulation timespan
    dt = 0.01 
    t_span = (0, 10)  # 20 seconds simulation
    Nsim = int(t_span[1]/dt)
    t_eval = np.linspace(t_span[0], t_span[1], Nsim)

    simU = np.zeros((Nsim, nu))     # Matrix to store the optimal control derivative
    simX = np.zeros((Nsim, nx))   # Matrix to store the simulated states

    # Declare the initial state
    x0 = np.array([1.469e+00, -4.231e-02,  4.699e-03, 1.000e+00, 0.000e+00,0.000e+00, 0.000e+00,    # eta
                   0.000e+00, 0.000e+00, -0.000e+00,  0.000e+00, 0.000e+00, 0.000e+00,              # nu
                   0.000e+00,  5.100e+01,  0.000e+00,  0.000e+00,  1.000e-06, 1.000e-06])           # control

    wp = np.array([6.613, -0.046, 0.656, 0.000, 0.000, 0.000, 1.000])
    ref = np.zeros((N_horizon, (nx+nu)))
    ref[:,:7] = wp

    simX[0,:] = x0

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
    for i in tqdm(range(Nsim-1)):

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
        if status != 0:
            print(f" Note: acados_ocp_solver returned status: {status}")

        # simulate system
        np.set_printoptions(precision=3)
        t[i] = ocp_solver.get_stats('time_tot')
        simU[i, :] = ocp_solver.get(0, "u")
        
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    data = np.concatenate([simX.T, simU.T])

    sol = Sol(t_eval,data)

    return sol


def plot_results(sol):
    """
    Plot simulation results.
    """

    def quaternion_to_euler_vec(sol):

        n = len(sol.y[3])
        psi = np.zeros(n)
        theta = np.zeros(n)
        phi = np.zeros(n)

        for i in range(n):
            q = [sol.y[3,i], sol.y[4,i], sol.y[5,i], sol.y[6,i]]
            psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)

        return psi, theta, phi

    psi_vec, theta_vec, phi_vec = quaternion_to_euler_vec(sol)

    _, axs = plt.subplots(8, 3, figsize=(12, 10))

    # Position plots
    axs[0,0].plot(sol.t, sol.y[0], label='x')
    axs[0,1].plot(sol.t, sol.y[1], label='y')
    axs[0,2].plot(sol.t, -sol.y[2], label='z')
    axs[0,0].set_ylabel('x Position [m]')
    axs[0,1].set_ylabel('y Position [m]')
    axs[0,2].set_ylabel('-z Position [m]')

    # Euler plots
    axs[1,0].plot(sol.t, np.rad2deg(phi_vec), label='roll')
    axs[1,1].plot(sol.t, np.rad2deg(theta_vec), label='pitch')
    axs[1,2].plot(sol.t, np.rad2deg(psi_vec), label='yaw')
    axs[1,0].set_ylabel('roll [deg]')
    axs[1,1].set_ylabel('pitch [deg]')
    axs[1,2].set_ylabel('yaw [deg]')

    # Velocity plots
    axs[2,0].plot(sol.t, sol.y[7], label='u')
    axs[2,1].plot(sol.t, sol.y[8], label='v')
    axs[2,2].plot(sol.t, sol.y[9], label='w')
    axs[2,0].set_ylabel('u (x_dot)')
    axs[2,1].set_ylabel('v (y_dot)')
    axs[2,2].set_ylabel('w (z_dot)')

    axs[3,0].plot(sol.t, sol.y[10], label='p')
    axs[3,1].plot(sol.t, sol.y[11], label='q')
    axs[3,2].plot(sol.t, sol.y[12], label='r')
    axs[3,0].set_ylabel('p (roll_dot)')
    axs[3,1].set_ylabel('q (pitch_dot)')
    axs[3,2].set_ylabel('r (yaw_dot)')

    # Control Input
    axs[4,0].plot(sol.t, sol.y[13], label='vbs')
    axs[4,1].plot(sol.t, sol.y[14], label='lcg')
    axs[4,2].plot(sol.t, sol.y[15], label='ds')
    axs[4,0].set_ylabel('vbs')
    axs[4,1].set_ylabel('lcg')
    axs[4,2].set_ylabel('ds')

    axs[5,0].plot(sol.t, sol.y[16], label='dr')
    axs[5,1].plot(sol.t, sol.y[17], label='rpm1')
    axs[5,2].plot(sol.t, sol.y[18], label='rpm2')
    axs[5,0].set_ylabel('dr')
    axs[5,1].set_ylabel('rpm1')
    axs[5,2].set_ylabel('rpm2')

    # Control derivatives
    axs[6,0].plot(sol.t, sol.y[19], label='dot_vbs')
    axs[6,1].plot(sol.t, sol.y[20], label='dot_lcg')
    axs[6,2].plot(sol.t, sol.y[21], label='dot_ds')
    axs[6,0].set_ylabel('dot_vbs')
    axs[6,1].set_ylabel('dot_lcg')
    axs[6,2].set_ylabel('dot_ds')

    axs[7,0].plot(sol.t, sol.y[22], label='dot_dr')
    axs[7,1].plot(sol.t, sol.y[23], label='dot_rpm1')
    axs[7,2].plot(sol.t, sol.y[24], label='dot_rpm2')
    axs[7,0].set_ylabel('dot_u_dr')
    axs[7,1].set_ylabel('dot_rpm1')
    axs[7,2].set_ylabel('dot_rpm2')

    plt.xlabel('Time [s]')
    plt.tight_layout()

def plot_trajectory(sol, numDataPoints, generate_gif=False, filename="3d.gif", FPS=10):
    
    # State vectors
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    
    # down-sampling the xyz data points
    N = y[::len(x) // numDataPoints];
    E = x[::len(x) // numDataPoints];
    D = z[::len(x) // numDataPoints];
    
    # Animation function
    def anim_function(num, dataSet, line):
        
        line.set_data(dataSet[0:2, :num])    
        line.set_3d_properties(dataSet[2, :num])    
        ax.view_init(elev=10.0, azim=-120.0)
        
        return line
    
    dataSet = np.array([N, E, -D])      # Down is negative z
    
    # Attaching 3D axis to the figure
    fig = plt.figure(2,figsize=(7,7))
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax) 
    
    # Line/trajectory plot
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='b')[0] 

    # Setting the axes properties
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlim3d([-20, 3])
    ax.set_zlabel('-Z / Down')

    # Plot 2D surface for z = 0
    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min-20, x_max+20)
    y_grid = np.arange(y_min-20, y_max+20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx
    ax.plot_surface(xx, yy, zz, alpha=0.3)
                    
    # Title of plot
    ax.set_title('North-East-Down')
    
    if generate_gif:
        # Create the animation object
        ani = animation.FuncAnimation(fig, 
                             anim_function, 
                             frames=numDataPoints, 
                             fargs=(dataSet,line),
                             interval=200, 
                             blit=False,
                             repeat=True)
        
        # Save the 3D animation as a gif file
        ani.save(filename, writer=animation.PillowWriter(fps=FPS))  


# Run simulation and plot results
#sol = run_simulation(t_span, x0, dt, sam)
sol = run_simulation()

plot_results(sol)
#plot_trajectory(sol, 50, False, "3d.gif", 10)
plt.show()

