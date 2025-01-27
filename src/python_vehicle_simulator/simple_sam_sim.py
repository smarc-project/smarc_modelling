import numpy as np
from scipy.integrate import solve_ivp
from python_vehicle_simulator.vehicles import *
from python_vehicle_simulator.lib import *
#from python_vehicle_simulator.vehicles.SAM import SAM
from python_vehicle_simulator.vehicles.SimpleSAM import SimpleSAM
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

# Initial conditions
eta0 = np.zeros(7)
eta0[3] = 1.0  # Initial quaternion (no rotation) 
nu0 = np.zeros(6)  # Zero initial velocities
u0 = np.zeros(6)
u0[0] = 50
u0[1] = 45
x0 = np.concatenate([eta0, nu0, u0])

# Simulation timespan
dt = 0.01 
t_span = (0, 30)  # 20 seconds simulation
n_sim = int(t_span[1]/dt)
t_eval = np.linspace(t_span[0], t_span[1], n_sim)

# Create SAM instance
sam = SimpleSAM(dt)

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
        u = np.zeros(6)
        u[0] = 50#*np.sin((i/(20/0.02))*(3*np.pi/4))        # VBS/
        u[1] = 50 # LCG
        #u[2] = np.deg2rad(7)    # Vertical (stern)
        u[3] = -np.deg2rad(7)   # Horizontal (rudder)
        u[4] = 1000     # RPM 1
        u[5] = u[4]     # RPM 2
        return sam.dynamics(x, u)

    # Run integration
    print(f" Start simulation")

    data = np.empty((len(x0), n_sim))
    data[:,0] = x0

    # Euler forward integration
    for i in range(n_sim-1):
        data[:,i+1] = data[:,i] + dynamics_wrapper(i, data[:,i]) * (t_span[1]/n_sim)
    sol = Sol(t_eval,data)
    print(f" Simulation complete!")

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
