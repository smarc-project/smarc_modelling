import numpy as np

from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.BlueROV import BlueROV
from smarc_modelling.vehicles.SAM import SAM

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

from scipy.spatial.transform import Rotation as R

matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

# Initial conditions
eta0 = np.zeros(7)
eta0[2] = 0
eta0[3] = 1.0  # Initial quaternion (no rotation) 
nu0 = np.zeros(6)  # Zero initial velocities
x0 = np.concatenate([eta0, nu0])

# Simulation timespan
dt = 0.01 #0.01 
t_span = (0, 10)  # 20 seconds simulation
n_sim = int(t_span[1]/dt)
t_eval = np.linspace(t_span[0], t_span[1], n_sim)

# ENU <-> NED conversion matrix
T = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])

# Create SAM instance
blueROV = BlueROV(dt)

class Sol():
    """
    Solver data class to match with Omid's plotting functions
    """
    def __init__(self, t, data) -> None:
        self.t = t
        self.y = data

        
def rk4(x, u, dt, fun):
    k1 = fun(x, u)
    k2 = fun(x+dt/2*k1, u)
    k3 = fun(x+dt/2*k2, u)
    k4 = fun(x+dt*k3, u)

    x_t = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return x_t

# FIXME: consider removing the dynamics wrapper and just call the dynamics straight away.
def run_simulation(t_span, x0, dt, blueROV):
    """
    Run BlueROV simulation using RK4.
    """
    u = np.zeros(6)
    u[0] =  10  # force in x-direction
    #u[1] = 50 # force in y-direction
    #u[2] = -10 # force in z-direction
    #u[3] = -1 # torque around x-axis
    #u[4] = -1 # torque around the y-axis
    #u[5] = -1 # torque around the z-axis

    nx = len(x0)
    nu = len(u)

    # Run integration
    print(f" Start simulation")

    data = np.empty((nx + nu, n_sim))
    data[:nx,0] = x0
    data[nx:,0] = u

    in_ENU = True
    frame_message_printed = False

    for i in range(n_sim-1):
        if in_ENU is True:
            if not frame_message_printed:
                print("You provide x0 and u in ENU")
                print("You get x and u in ENU")
                frame_message_printed = True

            pos_ned, quat_ned= enu_to_ned(data[:3,i], data[3:7,i])
            u_NED = u_enu_to_ned(u)

            x_NED = np.concatenate((pos_ned, quat_ned, data[7:nx,i]))

            x_new_NED = rk4(x_NED, u_NED, dt, blueROV.dynamics)
            pos_enu, quat_enu = ned_to_enu(x_new_NED[:3], x_new_NED[3:7])
            data[:3,i+1] = pos_enu
            data[3:7,i+1] = quat_enu
            data[7:nx,i+1] = x_new_NED[7:nx]
            data[nx:,i+1] = u
        else:
            if not frame_message_printed:
                print("You provide x0 and u in NED (default)")
                print("You get x and u in NED (default)")
                frame_message_printed = True
            data[:nx,i+1] = rk4(data[:nx,i], u, dt, blueROV.dynamics)
            data[nx:,i+1] = u
    sol = Sol(t_eval,data)
    print(f" Simulation complete!")

    return sol


def ned_to_enu(pos_ned, quat_ned):
    pos_enu = T @ pos_ned
    r_ned = R.from_quat(quat_ned)
    r_enu = T @ r_ned.as_matrix() @ T.T
    quat_enu = R.from_matrix(r_enu).as_quat()
    return pos_enu, quat_enu

def enu_to_ned(pos_enu, quat_enu):
    pos_ned = T @ pos_enu
    r_enu = R.from_quat(quat_enu)
    r_ned = T @ r_enu.as_matrix() @ T.T
    quat_ned = R.from_matrix(r_ned).as_quat()
    return pos_ned, quat_ned

def u_enu_to_ned(u):
    T = np.diag([1, -1, -1, 1, -1, -1])
    u_ned = T @ u
    return u_ned


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

    axs[4,0].plot(sol.t, sol.y[13], label='x')
    axs[4,1].plot(sol.t, sol.y[14], label='y')
    axs[4,2].plot(sol.t, sol.y[15], label='z')
    axs[4,0].set_ylabel('u_x')
    axs[4,1].set_ylabel('u_y')
    axs[4,2].set_ylabel('u_z')

    axs[5,0].plot(sol.t, sol.y[16], label='roll')
    axs[5,1].plot(sol.t, sol.y[17], label='pitch')
    axs[5,2].plot(sol.t, sol.y[18], label='yaw')
    axs[5,0].set_ylabel('u_roll')
    axs[5,1].set_ylabel('u_pitch')
    axs[5,2].set_ylabel('u_yaw')

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
sol = run_simulation(t_span, x0, dt, blueROV)
plot_results(sol)
plot_trajectory(sol, 50, False, "3d.gif", 10)
plt.show()
