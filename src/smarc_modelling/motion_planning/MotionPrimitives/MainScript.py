import smarc_modelling.motion_planning.MotionPrimitives.MapGeneration as MapGen
import numpy as np
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
import smarc_modelling.motion_planning.MotionPrimitives.MapGeneration as MapGen
from smarc_modelling.motion_planning.MotionPrimitives.GenerationTree import a_star_search, double_a_star_search, body_to_global_velocity, testOptimization
from smarc_modelling.motion_planning.MotionPrimitives.PlotResults import *
from smarc_modelling.motion_planning.MotionPrimitives.trm_colors import *
from smarc_modelling.motion_planning.MotionPrimitives.StatisticalAnalysis import runStatisticalAnalysis
import time
import matplotlib.animation as animation
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ckdtree

def MotionPlanningAlgorithm(realTimeDraw):
    """
    realTimeDraw: plot the results, True or False
    """

    # Generate and draw the map
    print("START")
    print(f"{bcolors.HEADER}>> Generate the map{bcolors.ENDC}")
    map_instance = MapGen.generationFirstMap()
    complexity = MapGen.evaluateComplexityMap(map_instance)
    print("complexity:",complexity)
    
    if realTimeDraw:
        ax, plt, fig = plot_map(map_instance, "top") # this is the one were the primitives are plotted
        ax2, plt2, fig2 = plot_map(map_instance, "side")
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    
    # Which cost function you want to use (1-Astar 2-AdaptiveAstar 3-HeadingConstraint) 
    typeFunction = 3

    # The tuning parameter in the algorithm (if using typeFunction=3)
    dec = 0.1

    # Search the path
    print(f"{bcolors.HEADER}>> Trajectory search{bcolors.ENDC}")
    start_time = time.time()
    if not realTimeDraw:
        ax = None 
        plt = None

    # Skip search or not
    onlyOptimization = False
    if not onlyOptimization:
        if complexity == 0:
            print(f"{bcolors.WARNING}single tree search{bcolors.ENDC}")
            trajectory, succesfulSearch, totalCost = a_star_search(ax, plt, map_instance, realTimeDraw, typeFunction, dec)
        else:
            print(f"{bcolors.WARNING}double tree search{bcolors.ENDC}")
            trajectory, succesfulSearch, totalCost = double_a_star_search(ax, plt, map_instance, realTimeDraw, typeFunction, dec)
    else:
        trajectory = testOptimization(map_instance)
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    end_time = time.time()

    # Computational time (seconds)
    print(f"{bcolors.HEADER}>> A star analysis{bcolors.ENDC}")
    print(f"total time for Astar:...{end_time-start_time:.4f} seconds")
    
    # Number of vertices in the trajectory
    print(f"length of trajectory: {len(trajectory):.2f} vertices>>")
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    # Save the trajectory into saved_trajectory.csv file
    print(f"{bcolors.HEADER}>> Save the trajectory >> saved_trajectory.csv{bcolors.ENDC}")
    df = pd.DataFrame(trajectory, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
    df.to_csv("saved_trajectory.csv", index=False)
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")

    # Draw SAM torpedo in the map 
    ind = 0
    if realTimeDraw:
        print(f"{bcolors.HEADER}>> Draw SAM as a cylinder in the plot{bcolors.ENDC}")
    for vertex in trajectory:

        # Print the velocity for each vertex in the trajectory
        q0, q1, q2, q3 = vertex[3:7]
        globalV = body_to_global_velocity((q0, q1, q2, q3), vertex[7:10])
        print(f"Velocity {ind:.0f} = {np.linalg.norm(globalV): .2f} m/s")

        # Case 4 ---TO CONTINUE
        '''
        v_CG_inertial = np.array([globalV[0], globalV[1], globalV[2]])        # Linear velocity of CG in inertial frame
        rr, ww, vv = vertex[10:13]
        omega_body = np.array([rr, ww, vv])              # Angular velocity in body frame
        r_fwd_body = np.array([0.655, 0, 0])           # Position of forward point relative to CG in body frame
        # Ensure quaternion is in (x, y, z, w) format for scipy
        rotation = R.from_quat([q1, q2, q3, q0])
        R_b2i = rotation.as_matrix()  # Body to inertial rotation matrix
        # Compute cross product in body frame
        v_relative_body = np.cross(omega_body, r_fwd_body)
        # Rotate to inertial frame
        v_relative_inertial = R_b2i @ v_relative_body
        # Add to CG velocity
        v_fwd_inertial = v_CG_inertial + v_relative_inertial
        #print(f" Velocity of pointA:", v_fwd_inertial)
        '''

        if realTimeDraw:
            # Draw torpedo in the two created plots
            norm_index = (ind / len(trajectory)) 
            #norm_index = 0.7
            draw_torpedo(ax, vertex, norm_index)
            draw_torpedo(ax2, vertex, norm_index)

            # Draw the velocity vector for each vertex (global velocity)
            x,y,z = vertex[:3]
            pointA = compute_A_point_forward(vertex)
            x_goal = map_instance["goal_pixel"][0]
            y_goal = map_instance["goal_pixel"][1]
            z_goal = map_instance["goal_pixel"][2]
            vx, vy, vz = globalV
            velocity_vector = np.array([vx, vy, vz])
            velocity_vector_norm = np.linalg.norm(velocity_vector)
            #ax.quiver(pointA[0], pointA[1], pointA[2], v_fwd_inertial[0], v_fwd_inertial[1], v_fwd_inertial[2], color='b', length=velocity_vector_norm, normalize=True)
            #ax.quiver(x, y, z, vx, vy, vz, color='b', length=velocity_vector_norm, normalize=True)
            #ax2.quiver(x, y, z, vx, vy, vz, color='b', length=velocity_vector_norm, normalize=True)

        ind += 1

    # Show the map, path and SAM 
    if realTimeDraw:
        plt.show()
        print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    
    # Create the gif
    gifOutput = True
    if gifOutput:
        print(f"{bcolors.HEADER}>> Generate GIF{bcolors.ENDC}")
        # Set up figure and 3D axis
        ax, plt, fig = plot_map(map_instance, "gif")

        # Create animation
        step = 10  # how many points to skip
        frame_indices = range(0, len(trajectory), step)
        ani = animation.FuncAnimation(fig, update, frames=frame_indices, fargs=(ax, plt, trajectory), interval=100)

        # Save as GIF
        ani.save('/home/parallels/Desktop/smarc_modelling-master/pics/torpedo_motion.gif', writer="ffmpeg", fps=5)

        # Show animation
        print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}THE END{bcolors.ENDC}")

def MotionPlanningROS(map_instance):
    """
    This is the function called by the ROS node. It takes:
    -)  map_instance as the input, as defined in the GenerationMap.py script
    and outputs:
    -)  a list of waypoints adnd a successful Flag
    """

    
    print("START")
    complexity = MapGen.evaluateComplexityMap(map_instance)
    print("complexity:",complexity)
    
    # Which cost function you want to use (1-Astar 2-AdaptiveAstar 3-HeadingConstraint) 
    typeFunction = 3

    # The tuning parameter in the algorithm (if using typeFunction=3)
    dec = 0.1

    # Search the path
    print(f"{bcolors.HEADER}>> Trajectory search{bcolors.ENDC}")
    start_time = time.time()
    if complexity == 0:
        print(f"{bcolors.WARNING}single tree search{bcolors.ENDC}")
        trajectory, succesfulSearch, totalCost = a_star_search(None, None, map_instance, False, typeFunction, dec)
    else:
        print(f"{bcolors.WARNING}double tree search{bcolors.ENDC}")
        trajectory, succesfulSearch, totalCost = double_a_star_search(None, None, map_instance, False, typeFunction, dec)
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    end_time = time.time()

    # Computational time (seconds)
    print(f"{bcolors.HEADER}>> A star analysis{bcolors.ENDC}")
    print(f"total time for Astar:...{end_time-start_time:.4f} seconds")
    
    # Number of vertices in the trajectory
    print(f"length of trajectory: {len(trajectory):.2f} vertices>>")
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")

    # Save the trajectory into saved_trajectory.csv file
    print(f"{bcolors.HEADER}>> Save the trajectory >> saved_trajectory.csv{bcolors.ENDC}")
    df = pd.DataFrame(trajectory, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
    df.to_csv("saved_trajectory.csv", index=False)
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    
    print(f"{bcolors.OKGREEN}THE END{bcolors.ENDC}")

    return (trajectory, succesfulSearch)

if __name__ == "__main__":

    #MotionPlanningAlgorithm(True)
    #runStatisticalAnalysis(1, 0)    # (nTrials, chosenComplexity)


    # Try it for the ROS package
    map_instance = MapGen.generationFirstMap()
    start_state = map_instance["initial_state"]
    goal_state = map_instance["final_state"]
    
    map_instance2 = MapGen.generateMapInstance(start_state, goal_state)
    trajectory, successfulFlag = MotionPlanningROS(map_instance2)
    draw_map_and_toredo(map_instance2, trajectory)
    ## Add if at least one tree arrives in the dataset