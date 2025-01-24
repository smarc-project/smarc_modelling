import numpy as np
from scipy.integrate import solve_ivp
from python_vehicle_simulator.vehicles import *
from python_vehicle_simulator.lib import *
#from python_vehicle_simulator.vehicles.SAM import SAM
from python_vehicle_simulator.vehicles.SimpleSAM import SimpleSAM
import matplotlib
import mpl_toolkits.mplot3d.axes3d as p3
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

# Initial conditions
eta0 = np.zeros(6)
#eta0[6] = 1.0  # Initial quaternion (no rotation) 
nu0 = np.zeros(6)  # Zero initial velocities
u0 = np.zeros(6)
u0[0] = 50
u0[1] = 50
x0 = np.concatenate([eta0, nu0, u0])

# Simulation timespan
n_sim = 300
t_span = (0, 10)  # 20 seconds simulation
t_eval = np.linspace(t_span[0], t_span[1], n_sim)
dt = t_span[1]/n_sim

# Create SAM instance
sam = SimpleSAM(dt)

class Sol():

    def __init__(self, t, data) -> None:
        self.t = t
        self.y = data



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
        data[5,i] = ((data[5,i] + np.pi)% (2*np.pi)) - np.pi
        data[:,i+1] = data[:,i] + dynamics_wrapper(i, data[:,i]) * (t_span[1]/n_sim)
    sol = Sol(t_eval,data)
    print(f" Simulation complete!")

    # RK 45 leads to numerical instabilities when setting the cb on top of the cg
#    sol = solve_ivp(
#        dynamics_wrapper,
#        t_span,
#        x0,
#        method='RK45',
#        t_eval=t_eval,
#        rtol=1e-6,
#        atol=1e-9
#    )
#    if sol.status == -1:
#        print(f" Simulation failed: {sol.message}")
#    else:
#        print(f" Simulation complete!")

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

    #psi_vec, theta_vec, phi_vec = quaternion_to_euler_vec(sol)
    phi_vec, theta_vec, psi_vec = sol.y[3,:], sol.y[4,:], sol.y[5,:] 

    fig, axs = plt.subplots(6, 3, figsize=(12, 10))

    # Position plots
    axs[0,0].plot(sol.t, sol.y[1], label='x')
    axs[0,1].plot(sol.t, sol.y[0], label='y')
    axs[0,2].plot(sol.t, -sol.y[2], label='z')
    axs[0,0].set_ylabel('x Position [m]')
    axs[0,1].set_ylabel('y Position [m]')
    axs[0,2].set_ylabel('-z Position [m]')
#    axs[0].legend()

    # Quaternion plots
#    axs[1,0].plot(sol.t, sol.y[3], label='q1')
#    axs[1,].plot(sol.t, sol.y[4], label='q2')
#    axs[1].plot(sol.t, sol.y[5], label='q3')
#    axs[1].plot(sol.t, sol.y[6], label='q0')
#    axs[1].set_ylabel('Quaternion')
#    axs[1].legend()

    # Euler plots
    axs[1,0].plot(sol.t, np.rad2deg(phi_vec), label='roll')
    axs[1,1].plot(sol.t, np.rad2deg(theta_vec), label='pitch')
    axs[1,2].plot(sol.t, np.rad2deg(psi_vec), label='yaw')
    axs[1,0].set_ylabel('roll [deg]')
    axs[1,1].set_ylabel('pitch [deg]')
    axs[1,2].set_ylabel('yaw [deg]')
#    axs[1].legend()

    # Velocity plots
    axs[2,0].plot(sol.t, sol.y[6], label='u')
    axs[2,1].plot(sol.t, sol.y[7], label='v')
    axs[2,2].plot(sol.t, sol.y[8], label='w')
    axs[2,0].set_ylabel('u (x_dot)')
    axs[2,1].set_ylabel('v (y_dot)')
    axs[2,2].set_ylabel('w (z_dot)')

    axs[3,0].plot(sol.t, sol.y[9], label='p')
    axs[3,1].plot(sol.t, sol.y[10], label='q')
    axs[3,2].plot(sol.t, sol.y[11], label='r')
    axs[3,0].set_ylabel('p (roll_dot)')
    axs[3,1].set_ylabel('q (pitch_dot)')
    axs[3,2].set_ylabel('r (yaw_dot)')

    axs[4,0].plot(sol.t, sol.y[12], label='vbs')
    axs[4,1].plot(sol.t, sol.y[13], label='lcg')
    axs[4,2].plot(sol.t, sol.y[14], label='dr')
    axs[4,0].set_ylabel('u_vbs')
    axs[4,1].set_ylabel('u_lcg')
    axs[4,2].set_ylabel('u_dr')

    axs[5,0].plot(sol.t, sol.y[15], label='ds')
    axs[5,1].plot(sol.t, sol.y[16], label='rpm1')
    axs[5,2].plot(sol.t, sol.y[17], label='rpm2')
    axs[5,0].set_ylabel('ds')
    axs[5,1].set_ylabel('rpm1')
    axs[5,2].set_ylabel('rpm2')
    #axs[3].legend()

#    # ksi plots
#    axs[4].plot(sol.t, sol.y[13], label='VBS')
#    axs[4].plot(sol.t, sol.y[14], label='LCG')
#    axs[4].plot(sol.t, sol.y[15], label='δs')
#    axs[4].plot(sol.t, sol.y[16], label='δr')
#    axs[4].plot(sol.t, sol.y[17], label='θ1')
#    axs[4].plot(sol.t, sol.y[18], label='θ2')
#    axs[4].set_ylabel('ksi')
#    axs[4].legend()
#
#    # ksi_dot plots
#    axs[5].plot(sol.t, sol.y[19], label='VBS')
#    axs[5].plot(sol.t, sol.y[20], label='LCG')
#    axs[5].plot(sol.t, sol.y[21], label='δs')
#    axs[5].plot(sol.t, sol.y[22], label='δr')
#    axs[5].plot(sol.t, sol.y[23], label='θ1')
#    axs[5].plot(sol.t, sol.y[24], label='θ2')
#    axs[5].set_ylabel('ksi_dot')
#    axs[5].legend()

    # ksi_ddot comparison
#    labels = ['VBS', 'LCG', 'δs', 'δr', 'θ1', 'θ2']
#    for i in range(6):
#        axs[6].plot(sol.t, ksi_ddot_unbounded[:, i], '--',
#                    label=f'{labels[i]} (unbounded)')
#        axs[6].plot(sol.t, ksi_ddot_bounded[:, i], '-',
#                    label=f'{labels[i]} (bounded)')
#    axs[6].set_ylabel('ksi_ddot')
#    axs[6].legend()

    plt.xlabel('Time [s]')
    plt.tight_layout()

def plot_trajectory(sol, numDataPoints, filename, FPS):
    
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
    ax.set_zlim3d([-20, 3])                   # default depth = -100 m
    
    #if np.amax(z) > 100.0:
    #    ax.set_zlim3d([-np.amax(z), 20])
        
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
    
#    # Create the animation object
#    ani = animation.FuncAnimation(fig, 
#                         anim_function, 
#                         frames=numDataPoints, 
#                         fargs=(dataSet,line),
#                         interval=200, 
#                         blit=False,
#                         repeat=True)
#    
#    # Save the 3D animation as a gif file
#    ani.save(filename, writer=animation.PillowWriter(fps=FPS))  


# Run simulation and plot results
sol = run_simulation(t_span, x0, sam)
plot_results(sol)
plot_trajectory(sol, n_sim, "3d.gif", 10)
plt.show()
