import numpy as np
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM import SAM
#from smarc_modelling.MotionPrimitives.MapGeneration_MotionPrimitives import map_instance, TILESIZE
import smarc_modelling.MotionPrimitives.MapGeneration_MotionPrimitives as MapGen
from smarc_modelling.MotionPrimitives.GenerationTree_MotionPrimitives import a_star_search, body_to_global_velocity
from smarc_modelling.MotionPrimitives.ObstacleChecker_MotionPrimitives import arrived
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import time 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.stats import norm
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

def plot_map(numDataPoints, map_data):
    # Extract map properties
    map_x_max = map_data["x_max"]   #number of column-wise
    map_y_max = map_data["y_max"]   #number of row-wise
    map_z_max = map_data["z_max"]
    obstacles = map_data["obstacleDict"]    #in the grid (row, column and z)
    start_pos = map_data["start_pos"]   #(x,y,z)
    goal_pixel = map_data["goal_pixel"] #(x,y,z)
    TILESIZE = map_data["TileSize"]

    # 3D plot setup
    fig = plt.figure(figsize=(50, 50))
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

    # Plot colored tiles
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
                        alpha=0.5
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
    #ax.view_init(elev=-20, azim=90, roll=180)  # Adjust camera angle
    ax.view_init(elev=75, azim=90)  # Adjust camera angle

    max_range = np.array([map_x_max, map_y_max, map_z_max]).max()  # Find the largest dimension
    mid_x = (map_x_max) / 2
    mid_y = (map_y_max) / 2
    mid_z = (map_z_max) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    #plt.draw()
    #plt.show()
    return (ax, plt)

def draw_torpedo(ax, vertex, length=1.5, radius=0.095, resolution=50):
    """Draws a torpedo-like shape (cylinder with hemispherical ends) at (x, y, z) with orientation from quaternion."""
    # Find the parameters
    x, y, z, q0, q1, q2, q3 = vertex[:7]

    # Create cylinder (torpedo body)
    theta = np.linspace(0, 2 * np.pi, resolution)
    x_cyl = np.linspace(-0.5, 0.5, resolution) * length  # Adjusting length
    theta, x_cyl = np.meshgrid(theta, x_cyl)
    
    y_cyl = radius * np.cos(theta)
    z_cyl = radius * np.sin(theta)
    
    # Create hemispherical caps
    phi = np.linspace(0, np.pi, resolution)
    theta_sphere = np.linspace(0, 2 * np.pi, resolution)
    phi, theta_sphere = np.meshgrid(phi, theta_sphere)
    
    '''
    # Front cap
    x_cap_front = radius * np.sin(phi) + 0.25  
    y_cap_front = radius * np.sin(phi) * np.cos(theta_sphere)
    z_cap_front = radius * np.sin(phi) * np.sin(theta_sphere)
    '''
    # Rear cap
    x_cap_rear = radius * np.sin(phi) - 0.85 
    y_cap_rear = radius * np.sin(phi) * np.cos(theta_sphere)
    z_cap_rear = radius * np.sin(phi) * np.sin(theta_sphere)
    
    
    # Convert quaternion to rotation matrix
    r = R.from_quat([q1, q2, q3, q0])  # SciPy uses [x, y, z, w] order
    rotation_matrix = r.as_matrix()
    
    # Apply rotation
    def transform_points(x, y, z):
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
        rotated_points = rotation_matrix @ points  # Matrix multiplication
        return rotated_points[0].reshape(x.shape), rotated_points[1].reshape(y.shape), rotated_points[2].reshape(z.shape)
    
    x_cyl, y_cyl, z_cyl = transform_points(x_cyl, y_cyl, z_cyl)
    #x_cap_front, y_cap_front, z_cap_front = transform_points(x_cap_front, y_cap_front, z_cap_front)
    x_cap_rear, y_cap_rear, z_cap_rear = transform_points(x_cap_rear, y_cap_rear, z_cap_rear)
    
    # Apply translation
    x_cyl += x
    y_cyl += y
    z_cyl += z
    '''
    x_cap_front += x
    y_cap_front += y
    z_cap_front += z
    '''
    x_cap_rear += x
    y_cap_rear += y
    z_cap_rear += z
    

    # Plot surfaces
    ax.plot_surface(x_cyl, y_cyl, z_cyl, color='y', alpha=0.8)
    #ax.plot_surface(x_cap_front, y_cap_front, z_cap_front, color='y', alpha=0.8)
    ax.plot_surface(x_cap_rear, y_cap_rear, z_cap_rear, color='k', alpha=0.8)

if __name__ == "__main__":

    # Initialize variables
    numberTrials = 100
    successfulTrials = 0
    realTimeMap = False  # you get the map only while searching (not only successful)
    plotFinalMaps = False   # you get the map at the end of the end (only successful)
    successfulTrajectories = []
    successfulMaps = []
    failedMaps = []
    successfulCosts = []
    successfulTimes = []
    # set this variable to decide which statistical analysis you want
    # 1 # Single analysis
    # 2 # Comparative analysis

    typeOfAnalysis = 2

    match typeOfAnalysis:
        case 1: 
            # --- SINGLE ANALYSIS ---
            # Start cycles
            dec = 0.9
            for trial in range(numberTrials):
                print(f"-->TRIAL NUMBER:-->{trial + 1:.2f}")
                print("...searching")
                map_instance = MapGen.generationFirstMap()
                if realTimeMap:
                    ax, plt = plot_map(2, map_instance)
                    start_time_aStar = time.time()
                    trajectory, succesfulSearch, totalCost = a_star_search(ax, plt, map_instance, realTimeMap, typeOfAnalysis, dec)
                    end_time_aStar = time.time()
                else:
                    ax = None
                    start_time_aStar = time.time()
                    trajectory, succesfulSearch, totalCost = a_star_search(ax, plt, map_instance, realTimeMap, typeOfAnalysis, dec)
                    end_time_aStar = time.time()

                if succesfulSearch:
                    successfulTrajectories.append(trajectory)
                    successfulMaps.append(map_instance)
                    successfulCosts.append(totalCost)
                    successfulTimes.append(end_time_aStar-start_time_aStar)
                else:
                    failedMaps.append(map_instance)

                successfulTrials += succesfulSearch
                print(f"length of trajectory: {len(trajectory):.2f} vertices")

                if realTimeMap:
                    for vertex in trajectory:
                        draw_torpedo(ax, vertex)

                        # Velocity direction
                        x,y,z = vertex[:3]
                        x_goal = map_instance["goal_pixel"][0]
                        y_goal = map_instance["goal_pixel"][1]
                        z_goal = map_instance["goal_pixel"][2]
                        q0, q1, q2, q3 = vertex[3:7]
                        vx, vy, vz = body_to_global_velocity((q0, q1, q2, q3), vertex[7:10])
                        velocity_vector = np.array([vx, vy, vz])
                        velocity_vector_norm = np.linalg.norm(velocity_vector)
                        ax.quiver(x, y, z, vx, vy, vz, color='b', length=velocity_vector_norm, normalize=True)

                        # Goal vector 
                        goal_vector = np.array([x_goal-x, y_goal-y, z_goal-z])
                        goal_vector_norm = np.linalg.norm(goal_vector)

                        # Orientation 
                        rotation = R.from_quat([q1, q2, q3, q0])
                        body_forward = np.array([1.0, 0.0, 0.0])
                        body_backward = np.array([-1.0, 0.0, 0.0])
                        forward_vector = rotation.apply(body_forward)
                        forward_vector_norm = np.linalg.norm(forward_vector)
                        forward_vector /= forward_vector_norm
                        backward_vector = rotation.apply(body_backward)
                        backward_vector_norm = np.linalg.norm(backward_vector)
                        backward_vector /= backward_vector_norm

                        # Compute the minimum angle with goal
                        cos_theta = np.dot(forward_vector, goal_vector) / (forward_vector_norm * goal_vector_norm)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angle_between_vectors = np.arccos(cos_theta)
                        if angle_between_vectors < np.deg2rad(90):
                            orientation_vector = forward_vector
                        else:
                            orientation_vector = backward_vector
                    plt.show()

                elif plotFinalMaps:
                    for ii in range(len(successfulTrajectories)):
                        corresponding_map = successfulMaps[ii]
                        ax, plt = plot_map(2, corresponding_map)
                        for vertex in successfulTrajectories[ii]:
                            draw_torpedo(ax, vertex)
                    plt.show()

            print("###########################")
            print("######### SUMMARY #########")
            # Number of runs
            print(f"-) Number of runs:...{numberTrials:.2f}")

            # Percentage of successfull executions 
            print(f"-) Successful percentage:...{(successfulTrials/numberTrials)*100:.2f} %")

            # List all the costs of successful executions
            print("-) Successful costs:-----------------------------")
            for cost in successfulCosts:
                print(f"   | LENGTH--> {cost:.4f} meters")
            print("   ----------------------------------------------")

            # Amount of time from calling the function a_star_search() to the output obtained
            print("-) Computational times (successful runs):--------")
            for t in successfulTimes:
                print(f"   | TIME--> {t:.4f} seconds")
            print("   ----------------------------------------------")

            # RTD analysis
            counts, bin_edges = np.histogram(successfulTimes, bins=30)
            normalized_counts = counts / numberTrials
            plt.bar(bin_edges[:-1], normalized_counts, width=np.diff(bin_edges), alpha=0.6, color='k', edgecolor='black', align='edge')

            #plt.hist(successfulTimes, bins=30, density=True, alpha=0.6, color='k', edgecolor='black')
            sorted_times = np.sort(successfulTimes)
            cumulative_prob = np.arange(1, len(successfulTimes) + 1) / numberTrials
            plt.plot(sorted_times, cumulative_prob, marker="o", linestyle="-", color="b", label="RTD")
            plt.xlabel("Execution Time (seconds)")
            plt.ylabel("Probability Density")
            plt.title("Run-Time Distribution (RTD)")
            plt.grid(True)
            plt.show()

            print("###########################")
            print("###########################")
        case 2:
            # --- COMPARATIVE ANALYSIS ---
            # Which analysis you want? 1-typefunction // 2-deceleration
            whichAnalysis = 2

            # Start cycles
            functionsToCompare = [1, 3]
            decelerationsToCompare = [0.01, 0.1, 1, 10]
            # Initialization dictionaries
            all_results = []
            successfulTimesDictionary_byTypeFunct = {}
            successfulTrialsDictionary = {}

            for trial in range(numberTrials):
                print(f"-->TRIAL NUMBER:-->{trial + 1:.2f}")
                map_instance = MapGen.generationFirstMap()

                match whichAnalysis:
                    case 1:
                        dec = 0.5
                        for typeFunct in functionsToCompare:
                            print(f"Searching with function:...{typeFunct:.1f}...")
                            ax = None
                            start_time = time.time()
                            trajectory, successfulSearch, totalCost = a_star_search(ax, plt, map_instance, False, typeFunct, dec)
                            end_time = time.time()

                            #if typeFunct not in successfulTimesDictionary_byTypeFunct:
                            #    successfulTimesDictionary_byTypeFunct[typeFunct] = []
                            #if typeFunct not in successfulTrialsDictionary:
                            #    successfulTrialsDictionary[typeFunct] = 0

                            # Store the results
                            computat_time = end_time - start_time
                            numberVertices = len(trajectory)
                            cost = totalCost
                            success = successfulSearch

                            all_results.append((typeFunct, success, computat_time, numberVertices, cost, dec))
                    case 2:
                        # Function to Use
                        typeFunct = 3

                        for dec in decelerationsToCompare:
                            print(f"Searching with deceleration:...{dec:.1f}...")
                            ax = None
                            start_time = time.time()
                            trajectory, successfulSearch, totalCost = a_star_search(ax, plt, map_instance, False, typeFunct, dec)
                            end_time = time.time()

                            # Store the results
                            computat_time = end_time - start_time
                            numberVertices = len(trajectory)
                            cost = totalCost
                            success = successfulSearch

                            all_results.append((typeFunct, success, computat_time, numberVertices, cost, dec))

            print("################################################")
            print("######### SUMMARY COMPARATIVE ANALYSIS #########")
            # Number of runs
            #print(f"-) Number of runs:...{numberTrials:.2f}")

            # Percentage of successfull executions
            #for typeFun in functionsToCompare:
            #    print(f"-) Successful percentage function {typeFun:.1f}:...{(successfulTrialsDictionary[typeFun]/numberTrials)*100:.2f} %")

            # RTD analysis
            '''
            for typefunction in functionsToCompare:
                
                # Density distribution
                data = successfulTimesDictionary_byTypeFunct[typefunction]
                counts, bin_edges = np.histogram(data, bins=30)
                normalized_counts = counts / numberTrials
                plt.bar(bin_edges[:-1], normalized_counts, width=np.diff(bin_edges), alpha=0.6, color='k', edgecolor='black', align='edge')

                # Cumulative probability
                sorted_times = np.sort(data)
                cumulative_prob = np.arange(1, len(data) + 1) / numberTrials
                plt.plot(sorted_times, cumulative_prob, marker="o", linestyle="-", color="k", label="RTD")
                plt.xlabel("Execution Time (seconds)")
                plt.ylabel("Cumulative probability distribution")
                plt.title(f"Run-Time Distribution (RTD) for typeFunction {typefunction}")
                plt.grid(True)
                #plt.show()

                # Plot normal distribution
                mu = np.mean(data)
                sigma = np.std(data)
                x = np.linspace(min(data), max(data), 1000)
                pdf = norm.pdf(x, mu, sigma)
                plt.plot(x, pdf, 'b-', lw=2, label="Normal Distribution")
                plt.legend()
                plt.show()
            '''

            # Storing data in dataset
            df = pd.DataFrame(all_results, columns=["typeFunct", "success", "execution_time", "numberOfVertices", "total_cost", "deceleration"])
            df.to_csv("a_star_results.csv", index=False)
            print("Results are stored in a_star_results.csv file!")

            print("###########################")
            print("###########################")
        case _:
            print("No case specified!")