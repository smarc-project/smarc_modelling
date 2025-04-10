'''
Script to test and verify different models of SAM
'''
import sys
import os
# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import numpy as np
from control import NMPC
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM import SAM
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
from smarc_modelling.vehicles.SAM_LQR import SAM_LQR
from smarc_modelling.control.LQR_VSI import *



import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

# Initial conditions
eta0 = np.zeros(7)
eta0[3] = 1.0  # Initial quaternion (no rotation) 
nu0 = np.zeros(6)  # Zero initial velocities
nu0[0] = 1e-6
u0 = np.zeros(6)
u0[4:] = 10
u0[0] = 0
u0[1] = 0
x0 = np.concatenate([eta0, nu0, u0])

# Simulation timespan
dt = 0.1
t_span = (0, 30)  # 20 seconds simulation
n_sim = 102#int(t_span[1]/dt)
t_eval = np.linspace(t_span[0], t_span[1], n_sim)

# Create SAM instances
sam = SAM(dt)
sam_casadi = SAM_casadi(dt)
sam_dynamics = sam_casadi.dynamics(export=True)
samlqr = SAM_LQR(dt)
lqr_dynamics = samlqr.dynamics(export=False)   # The LQR model to be used.

lqr = LQR_TEST(SAM_casadi(dt).dynamics(export=True), dt)

class Sol():
    """
    Solver data class to match with Omid's plotting functions
    """
    def __init__(self, t, data) -> None:
        self.t = t
        self.y = data


# FIXME: consider removing the dynamics wrapper and just call the dynamics straight away.
def run_simulation(t_span, x0, sam):
    """
    Run SAM simulation using solve_ivp.
    """
    def dynamics_wrapper(t, x):
        """
        u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        """
        u = np.array([
            [50, 50, 0, 0, 0.001, 0.001],
            [49.968, 48.557, 0, 0.087, 400.01, 399.991],
            [49.977, 47.037, 0, 0.087, 400.007, 399.994],
            [49.984, 45.526, 0, 0.087, 400.007, 399.994],
            [49.987, 44.023, 0, 0.087, 400.009, 399.992],
            [49.988, 42.521, 0, 0.087, 400.008, 399.993],
            [49.99, 41.016, 0, 0.087, 399.998, 400.002],
            [49.996, 39.506, 0, 0.087, 399.994, 400.005],
            [50.004, 37.993, 0, 0.087, 399.991, 400.008],
            [50.014, 36.474, 0, 0.087, 399.981, 400.016],
            [50.024, 34.956, 0, 0.087, 399.959, 400.039],
            [50.032, 33.44, 0, 0.087, 399.975, 400.022],
            [50.036, 31.925, 0.001, 0.087, 399.979, 400.017],
            [50.039, 30.413, 0.001, 0.087, 399.981, 400.012],
            [50.044, 28.905, 0.001, 0.087, 399.986, 400.004],
            [50.048, 27.401, 0.002, 0.087, 399.987, 400],
            [50.05, 25.899, 0.002, 0.087, 399.981, 400.003],
            [50.051, 24.398, 0.002, 0.087, 399.976, 400.006],
            [50.053, 22.898, 0.002, 0.087, 399.967, 400.013],
            [50.054, 21.4, 0.002, 0.087, 399.961, 400.018],
            [50.054, 19.902, 0.002, 0.087, 399.953, 400.024],
            [50.05, 18.402, 0.002, 0.087, 399.939, 400.037],
            [50.046, 16.902, 0.002, 0.087, 399.935, 400.039],
            [50.045, 15.403, 0.002, 0.087, 399.934, 400.038],
            [50.045, 13.904, 0.002, 0.087, 399.921, 400.049],
            [50.048, 12.407, 0.002, 0.087, 399.912, 400.056],
            [50.05, 10.911, 0.002, 0.087, 399.9, 400.067],
            [50.049, 9.415, 0.002, 0.087, 399.886, 400.08],
            [50.047, 7.92, 0.002, 0.087, 399.886, 400.079],
            [50.045, 6.427, 0.002, 0.086, 399.889, 400.075],
            [51.051, 4.967, -0.132, -0.116, -200.226, -199.921],
            [52.044, 3.467, -0.131, -0.115, -200.253, -199.903],
            [53.041, 1.966, -0.131, -0.114, -200.302, -199.869],
            [54.039, 0.465, -0.131, -0.114, -200.335, -199.854],
            [55.046, -0.034, -0.131, -0.114, -200.471, -199.728],
            [56.051, -0.029, -0.13, -0.114, -200.533, -199.66],
            [57.049, -0.024, -0.129, -0.112, -200.421, -199.756],
            [58.046, -0.018, -0.128, -0.11, -200.4, -199.764],
            [59.042, -0.013, -0.127, -0.108, -200.532, -199.626],
            [60.033, -0.009, -0.126, -0.108, -200.401, -199.768],
            [61.022, -0.009, -0.126, -0.108, -200.455, -199.742],
            [62.015, -0.011, -0.126, -0.107, -200.546, -199.683],
            [63.013, -0.016, -0.126, -0.106, -200.668, -199.602],
            [64.025, -0.016, -0.126, -0.107, -200.8, -199.485],
            [65.04, -0.011, -0.126, -0.107, -201, -199.275],
            [66.05, -0.004, -0, -0.121, -1000.942, -999.164],
            [67.044, 0.007, -0, -0.121, -1000.834, -999.265],
            [68.036, 0.018, 0, -0.121, -1000.725, -999.375],
            [69.013, 0.023, 0, -0.121, -1000.758, -999.353],
            [69.995, 0.021, 0, -0.121, -1000.791, -999.337],
            [70.986, 0.016, 0, -0.121, -1001.054, -999.098],
            [71.989, 0.016, 0, -0.121, -1001.012, -999.175],
            [73.008, 0.027, 0, -0.121, -1001.003, -999.225],
            [73.993, 0.028, 0, -0.122, -1001.018, -999.266],
            [74.971, 0.026, 0, -0.122, -1001.078, -999.276],
            [75.939, 0.019, 0, -0.122, -1001.233, -999.214],
            [76.894, 0.004, 0, -0.122, -1001.565, -999.013],
            [77.835, -0.019, 0, -0.122, -1002.199, -998.568],
            [78.755, -0.051, 0, -0.122, -1003.289, -997.766],
            [79.63, -0.099, 0, -0.122, -1004.784, -996.709],
            [80.426, -0.174, 0, -0.121, -1006.345, -995.724],
            [81.2, -0.254, 0, -0.121, -1007.95, -994.732],
            [82.804, 0.067, 0.002, 0.037, -1011.624, -992.012],
            [83.397, -0.07, 0.002, 0.039, -1012.125, -991.689],
            [84.324, -0.107, 0.001, 0.04, -1012.635, -990.564],
            [85.436, -0.095, 0.001, 0.041, -1013.449, -989.025],
            [86.557, -0.091, 0, 0.041, -1014.333, -987.513],
            [87.669, -0.091, 0, 0.042, -1015.397, -985.942],
            [88.781, -0.086, -0, 0.043, -1016.713, -984.216],
            [89.905, -0.063, -0, 0.044, -1018.285, -982.308],
            [90.044, -0.014, 0, 0.045, -1020.007, -980.308],
            [90.177, 0.06, 0.001, 0.046, -1021.882, -978.207],
            [90.24, 0.133, 0.002, 0.048, -1024.055, -975.873],
            [90.153, 0.154, 0.003, 0.05, -1026.817, -973.029],
            [89.923, 0.098, 0.004, 0.052, -1030.58, -969.244],
            [89.67, -0.011, 0.003, 0.054, -1035.908, -963.897],
            [89.493, -0.149, 0.002, 0.057, -1043.193, -956.545],
            [89.442, -0.3, -0, 0.06, -1052.379, -947.207],
            [89.658, -0.389, -0.003, 0.064, -1060.163, -939.147],
            [90.529, -0.193, -0.005, 0.069, -1067.686, -931.08],
            [91.807, 0.322, -0.003, 0.074, -1078.473, -919.563],
            [93.073, 1.143, 0.004, 0.079, -1094.168, -903.105],
            [93.747, 2.134, 0.018, 0.086, -1106.716, -890.027],
            [91.553, 2.062, 0.03, 0.097, -1113.56, -883.639],
            [87.866, 0.715, 0.028, 0.107, -1124.733, -873.129],
            [85.754, -0.832, 0.011, 0.12, -1150.011, -847.577],
            [84.697, -2.682, -0.017, 0.135, -1184.314, -812.138],
            [85.872, -4.065, -0.045, 0.155, -1195.264, -798.811],
            [92.883, -2.733, -0.056, 0.177, -1209.374, -780.239],
            [90.344, -0.202, -0.636, 0.101, -226.683, 6.464],
            [87.937, 5.296, -1.902, 1.262, 53.254, 266.901],
            [90.721, 15.265, -2.762, 3.181, 59.666, 268.96],
            [96.884, 30.316, 1.648, -5.741, 94.166, 251.923],
            [92.759, 18.405, 1.946, -1.873, 76.743, 286.472],
            [92.425, 14.036, 1.424, -0.976, 73.147, 306.6],
            [93.216, 11.099, 1.138, -0.715, 53.748, 344.82],
            [93.578, 7.47, 0.965, -0.571, 76.981, 344.485],
            [90.85, 2.803, 0.867, -0.458, 107.947, 340.035],
            [89.752, -1.97, 0.81, -0.358, 128.032, 350.92],
            [90.58, -5.951, 0.759, -0.27, 139.796, 374.011],
            [91.845, -8.818, 0.7, -0.192, 163.723, 388.029],
            [95.717, -9.193, 0.626, -0.132, 211.123, 382.062],
            [95.252, -8.323, 0.577, -0.141, 255.212, 427.226]
        ])

        # Check the shape of the matrix

        # u = np.zeros(6)
        # u[0] = 50#*np.sin((i/(20/0.02))*(3*np.pi/4))        # VBS
        # u[1] = 50 # LCG
        # #u[2] = np.deg2rad(7)    # Vertical (stern)
        # #u[3] = -np.deg2rad(7)   # Horizontal (rudder)
        # u[4] = 1000     # RPM 1
        # u[5] = u[4]     # RPM 2

        # choose between numpy model (0) or casadi model (1)
        model = 2
        if model == 0:
            # Numpy SAM
            return sam.dynamics(x, u[t, :])
        elif model == 1:
            # SAM LQR
            x = np.delete(x,3)

            #x_dot = lqr_dynamics(x[:12],u) # Export True
            x_dot = lqr_dynamics(x,u[t, :])  # Export False

            q1 = x_dot[3]
            q2 = x_dot[4]
            q3 = x_dot[5]
            q0 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
            q = np.array([q0, q1, q2, q3]).flatten()
            x_dot = np.array(x_dot).flatten()
            #x_dot = np.hstack((x_dot, u))
            x_dot = np.hstack((x_dot[:3], q, x_dot[6:]))
            return x_dot
        else:
            # CASADI SAM
            x_dot = sam_dynamics(x[:13],u[t, :])
            x_dot = np.array(x_dot).flatten()

            return x_dot, u[t, :]

    # Run integration
    print(f" Start simulation")

    data = np.empty((len(x0), n_sim))
    data[:,0] = x0

    # Euler forward integration
    # NOTE: This integrates eta, nu, u_control in the same time step.
    #   Depending on the maneuvers, we might want to integrate nu and u_control first
    #   and use these to compute eta_dot. This needs to be determined based on the 
    #   performance we see.
    start_time = time.time()
    for i in range(n_sim-1):
        print(i)
        x_dot, u = dynamics_wrapper(i, data[:,i])
        x_dot = x_dot*dt
        x_dot = np.concatenate((x_dot, u-data[13:,i]))
        data[:,i+1] = data[:,i] + x_dot
        print(data[3:7 ,i])
    sol = Sol(t_eval,data)

    end_time = time.time()
    print(f" Simulation complete!")
    print(f"Time for simulation: {end_time-start_time}s")
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

    _, axs = plt.subplots(6, 3, figsize=(12, 10))

    # Position plots
    axs[0,0].plot(sol.t, sol.y[1], label='x')
    axs[0,1].plot(sol.t, sol.y[0], label='y')
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

    axs[4,0].plot(sol.t, sol.y[13], label='vbs')
    axs[4,1].plot(sol.t, sol.y[14], label='lcg')
    axs[4,2].plot(sol.t, sol.y[15], label='ds')
    axs[4,0].set_ylabel('u_vbs')
    axs[4,1].set_ylabel('u_lcg')
    axs[4,2].set_ylabel('u_ds')

    axs[5,0].plot(sol.t, sol.y[16], label='dr')
    axs[5,1].plot(sol.t, sol.y[17], label='rpm1')
    axs[5,2].plot(sol.t, sol.y[18], label='rpm2')
    axs[5,0].set_ylabel('u_dr')
    axs[5,1].set_ylabel('rpm1')
    axs[5,2].set_ylabel('rpm2')

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
sol = run_simulation(t_span, x0, sam)
plot_results(sol)
plot_trajectory(sol, 50, False, "3d.gif", 10)
plt.show()
