### To Do List ###
# 0. v/orientation in InputPair (GenTree), getNeigh (GenTree), calculate_f (GenTree) 
# 3. Find right decelleration

# angles: get_neigh, curvePrimitives

import numpy as np
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM import SAM
#from smarc_modelling.MotionPrimitives.MapGeneration_MotionPrimitives import map_instance, TILESIZE
import smarc_modelling.MotionPrimitives.MapGeneration_MotionPrimitives as MapGen
from smarc_modelling.MotionPrimitives.GenerationTree_MotionPrimitives import a_star_search, body_to_global_velocity
from smarc_modelling.MotionPrimitives.ObstacleChecker_MotionPrimitives import compute_A_point_forward, compute_B_point_backward
from smarc_modelling.MotionPrimitives.PostProcessing_MotionPrimitives import increaseResolutionTrajectory
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
from scipy.spatial.transform import Rotation as R
import time 
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

# Initial conditions for simulating ORIGINAL ONE (not primitives)
eta0 = np.zeros(7)
eta0[3] = 1  # Initial quaternion (no rotation) 
nu0 = np.zeros(6)  # Zero initial velocities
u0 = np.zeros(6)    #The initial control inputs for SAM
u0[0] = 50  #Vbs
u0[1] = 50  #lcg
x0 = np.concatenate([eta0, nu0, u0])

# Simulation timespan (not used, modify in GenerationPrimitives_MotionPrimitives.py if needed)
dt = 0.1
t_span = (0, 10)  # 20 seconds simulation
n_sim = int(t_span[1]/dt)   #number of runs__f
t_eval = np.linspace(t_span[0], t_span[1], n_sim)   #the vector of times

# Create SAM instance
sam = SAM(dt)

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
    def dynamics_wrapper(t, x): #Not used (modify in MotionPrimitives if needed)
        """
        u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        """
        u = np.zeros(6)
        u[0] = 0   # VBS
        u[1] = 50   # LCG
        u[2] = np.deg2rad(-7)    # Vertical (stern)--aileron // IN RADIANTS
        #u[3] = -np.deg2rad(7)   # Horizontal (rudder) // IN RADIANTS
        u[4] = 1000     # RPM 1
        u[5] = u[4]     # RPM 2
        return sam.dynamics(x, u)

    # Run integration
    print(f" Start simulation")

    data = np.empty((len(x0), n_sim))   #Amatrix containing for each state in x0, n_sim values (empty rn)
    #print(data.shape)
    data[:,0] = x0  #initial conditions written in the matrix

    # Euler forward integration
    # NOTE: This integrates eta, nu, u_control in the same time step.
    #   Depending on the maneuvers, we might want to integrate nu and u_control first
    #   and use these to compute eta_dot. This needs to be determined based on the 
    #   performance we see.
    for i in range(n_sim-1):
        data[:,i+1] = data[:,i] + dynamics_wrapper(i, data[:,i]) * (t_span[1]/n_sim)    #Should we edit it into dynamics_wrapper(i*t_span[1]/n_sim, data[:,i])????
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

def plot_map(map_data, typePlot):
    """
    This function creates the map to be shown. It colors the cells, create the tank and set the initial view with the axis limit.
    """

    # Extract map properties
    map_x_max = map_data["x_max"]   #number of column-wise
    map_y_max = map_data["y_max"]   #number of row-wise
    map_z_max = map_data["z_max"]
    obstacles = map_data["obstacleDict"]    #in the grid (row, column and z)
    start_pos = map_data["start_pos"]   #(x,y,z)
    goal_pixel = map_data["goal_pixel"] #(x,y,z)
    TILESIZE = map_data["TileSize"]
    (restr_x_min, restr_y_min, restr_z_min), (restr_x_max, restr_y_max, restr_z_max) = map_data["restricted_area"]

    # 3D plot setup
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot map as 3D surface 
    x_grid = np.arange(0, map_x_max, TILESIZE)
    y_grid = np.arange(0, map_y_max, TILESIZE)
    z_grid = np.arange(0, map_z_max, TILESIZE)
    xx, yy , zz = np.meshgrid(x_grid, y_grid, z_grid)   # Shape: (numberVertical, numberHorizontal, number3D) == (y,x,z)

    # Create color grid for map visualization
    color_grid = np.full_like(xx, "white", dtype=object)    #(rows, columns, z)
    for obs in obstacles:
        color_grid[obs[0], obs[1], obs[2]] = "black"
    color_grid[int(start_pos[1] // TILESIZE), int(start_pos[0] // TILESIZE), int(start_pos[2] // TILESIZE)] = "green"
    color_grid[int(goal_pixel[1] // TILESIZE), int(goal_pixel[0] // TILESIZE), int(goal_pixel[2] // TILESIZE)] = "red"

    plotRestrictedArea = False
    if plotRestrictedArea:
        # Transform restricted area from boundaries to cells
        restr_c_min_idx = int(restr_x_min // TILESIZE)
        restr_c_max_idx = int(restr_x_max // TILESIZE)
        restr_r_min_idx = int(restr_y_min // TILESIZE)
        restr_r_max_idx = int(restr_y_max // TILESIZE)
        restr_z_min_idx = int(restr_z_min // TILESIZE)
        restr_z_max_idx = int(restr_z_max // TILESIZE)

        # Define restricted area in color_grid
        for k in np.arange(restr_z_min_idx, restr_z_max_idx):  # z-axis
            for i in np.arange(restr_r_min_idx, restr_r_max_idx):  # row-axis (y)
                for j in np.arange(restr_c_min_idx, restr_c_max_idx):  # column-axis (x)
                    color_grid[i, j, k] = "cyan"  # Mark as restricted

    # Plot colored tiles from color_grid
    for k in range(color_grid.shape[2]):  # z-axis
        for i in range(color_grid.shape[0]):  # row-axis
            for j in range(color_grid.shape[1]):  # column-axis
                if color_grid[i,j,k] == "black" or color_grid[i,j,k] == "green" or color_grid[i,j,k] == "red":
                    ax.bar3d(
                        xx[i, j, k], 
                        yy[i, j, k],
                        zz[i, j, k],
                        TILESIZE, TILESIZE, TILESIZE,
                        color=color_grid[i, j, k],
                        edgecolor='gray',
                        alpha=0.9
                    )
                elif color_grid[i,j,k] == "cyan":
                    ax.bar3d(
                        xx[i, j, k], 
                        yy[i, j, k],
                        zz[i, j, k],
                        TILESIZE, TILESIZE, TILESIZE,
                        color="red",
                        edgecolor="none",
                        alpha=0.01
                    )

    # Plot start and goal points
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='green', marker='o', s=100, label="Start")
    ax.scatter(goal_pixel[0], goal_pixel[1], goal_pixel[2], c='red', marker='x', s=100, label="End")

    # Axis labels
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlabel('-Z / Down')
    ax.set_title('Trajectory with Map')
    ax.legend()

    # Decide which view you want
    match typePlot:
        case "top":
            ax.view_init(elev=75, azim=90)  # top view

        case "side":
            ax.view_init(elev=0, azim=0)  # right view

        case "rotated":
            ax.view_init(elev=25, azim=80)  # top view

        case _:
            ax.view_init(elev=75, azim=90)  # top view

    # Set the limits for the axis
    ax.set_xlim(0, map_x_max)
    ax.set_ylim(0, map_y_max)
    ax.set_zlim(0, map_z_max)
    ax.set_xlim(0, map_x_max)
    ax.set_ylim(0, map_y_max)
    ax.set_zlim(0, map_z_max)
    ax.set_box_aspect([map_x_max, map_y_max, map_z_max])

    # return the plot setup
    return (ax, plt)

def draw_torpedo(ax, vertex, colorr, length=1.5, radius=0.095, resolution=50):
    """
    Draws a torpedo-like shape (cylinder) and a black actuator at the back (disk) at (x, y, z) with orientation from quaternion.
    """

    # Find the parameters
    x, y, z, q0, q1, q2, q3 = vertex[:7]

    # Create cylinder (torpedo body)
    theta = np.linspace(0, 2 * np.pi, resolution)
    x_cyl = np.linspace(-0.5, 0.5, resolution) * length  # adjusting length
    theta, x_cyl = np.meshgrid(theta, x_cyl)
    y_cyl = radius * np.cos(theta)
    z_cyl = radius * np.sin(theta)
    
    # Create hemispherical caps
    r_disk = np.linspace(0, radius, resolution)  # Radial distances
    theta_disk = np.linspace(0, 2 * np.pi, resolution)  # Angles
    r_disk, theta_disk = np.meshgrid(r_disk, theta_disk)
    x_cap_rear = np.full_like(r_disk, -0.5 * length)  # Fixed x position (rear end of the torpedo)
    y_cap_rear = r_disk * np.cos(theta_disk)
    z_cap_rear = r_disk * np.sin(theta_disk)

    # Convert quaternion to rotation matrix
    r = R.from_quat([q1, q2, q3, q0]) 
    rotation_matrix = r.as_matrix()
    
    # Apply rotation
    def transform_points(x, y, z):
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
        rotated_points = rotation_matrix @ points  # Matrix multiplication
        return rotated_points[0].reshape(x.shape), rotated_points[1].reshape(y.shape), rotated_points[2].reshape(z.shape)
    x_cyl, y_cyl, z_cyl = transform_points(x_cyl, y_cyl, z_cyl)
    x_cap_rear, y_cap_rear, z_cap_rear = transform_points(x_cap_rear, y_cap_rear, z_cap_rear)
    
    # Apply translation
    x_cyl += x
    y_cyl += y
    z_cyl += z
    x_cap_rear += x 
    y_cap_rear += y
    z_cap_rear += z
    
    # Plot spheres
    plotSpheres = False
    if plotSpheres:
        pointA = compute_A_point_forward(vertex)
        pointB = compute_B_point_backward(vertex)
        radius = 0.095
        u = np.linspace(0, 2 * np.pi, 30)  # azimuthal angle
        v = np.linspace(0, np.pi, 30)      # polar angle

        # Convert spherical to Cartesian coordinates
        XA = radius * np.outer(np.cos(u), np.sin(v)) + pointA[0]
        YA = radius * np.outer(np.sin(u), np.sin(v)) + pointA[1]
        ZA = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pointA[2]
        XB = radius * np.outer(np.cos(u), np.sin(v)) + pointB[0]
        YB = radius * np.outer(np.sin(u), np.sin(v)) + pointB[1]
        ZB = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pointB[2]

        # Plot the sphere
        ax.plot_surface(XA, YA, ZA, color='k', alpha=0.6, edgecolor='k')
        ax.plot_surface(XB, YB, ZB, color='k', alpha=0.6, edgecolor='k')
    
    # Plot surfaces (cylinder and cap)
    ax.plot_surface(x_cyl, y_cyl, z_cyl, color='y', alpha=colorr)
    ax.plot_surface(x_cap_rear, y_cap_rear, z_cap_rear, color='k', alpha=colorr)

if __name__ == "__main__":

    # Run simulation and plot results
    '''    
    sol = run_simulation(t_span, x0, sam)
    plot_results(sol)
    plot_trajectory(sol, 50, True, "3d.gif", 10)'''

    
    # Generate and draw the map
    map_instance = MapGen.generationFirstMap()
    ax, plt = plot_map(map_instance, "top") # this is the one were the primitives are plotted
    ax2, plt2 = plot_map(map_instance, "side")

    # Which cost function you want to use (1-Astar 2-AdaptiveAstar 3-HeadingConstraint) 
    typeFunction = 3

    # The tuning parameter in the algorithm (if using typeFunction=3)
    dec = 0.1

    # Search the path
    start_time = time.time()
    trajectory, succesfulSearch, totalCost = a_star_search(ax, plt, map_instance, True, typeFunction, dec)
    end_time = time.time()

    # Post Processing
    maximumDistanceWaypoints = 0.5    # meters
    results = increaseResolutionTrajectory(trajectory, maximumDistanceWaypoints)
    print(results)

    # Computational time (seconds)
    print(f"total time for Astar:...{end_time-start_time:.4f} seconds")
    
    # Number of vertices in the trajectory
    print(f"length of trajectory: {len(trajectory):.2f} vertices")

    # Save the trajectory into saved_trajectory.csv file
    df = pd.DataFrame(trajectory, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
    df.to_csv("saved_trajectory.csv", index=False)

    # Draw SAM torpedo in the map 
    ind = 0
    for vertex in trajectory:

        # Print the velocity for each vertex in the trajectory
        q0, q1, q2, q3 = vertex[3:7]
        globalV = body_to_global_velocity((q0, q1, q2, q3), vertex[7:10])
        print(f"Velocity {ind:.0f} = {np.linalg.norm(globalV): .2f} m/s")

        # Draw torpedo in the two created plots
        norm_index = (ind / len(trajectory)) 
        draw_torpedo(ax, vertex, norm_index)
        draw_torpedo(ax2, vertex, norm_index)
        ind += 1

        # Draw the velocity vector for each vertex (global velocity)
        x,y,z = vertex[:3]
        x_goal = map_instance["goal_pixel"][0]
        y_goal = map_instance["goal_pixel"][1]
        z_goal = map_instance["goal_pixel"][2]
        vx, vy, vz = globalV
        velocity_vector = np.array([vx, vy, vz])
        velocity_vector_norm = np.linalg.norm(velocity_vector)
        ax.quiver(x, y, z, vx, vy, vz, color='b', length=velocity_vector_norm, normalize=True)
        ax2.quiver(x, y, z, vx, vy, vz, color='b', length=velocity_vector_norm, normalize=True)

    # Show the map, path and SAM 
    plt.show()