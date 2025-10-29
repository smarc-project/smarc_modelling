# import smarc_modelling.motion_planning.MotionPrimitives.MapGeneration as MapGen
import numpy as np
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
import smarc_modelling.motion_planning.MotionPrimitives.MapGeneration as MapGen
from smarc_modelling.motion_planning.MotionPrimitives.GenerationTree import a_star_search, double_a_star_search, body_to_global_velocity
from smarc_modelling.motion_planning.MotionPrimitives.PlotResults import *
from smarc_modelling.motion_planning.MotionPrimitives.trm_colors import *
from smarc_modelling.motion_planning.MotionPrimitives.StatisticalAnalysis import runStatisticalAnalysis
#from smarc_modelling.sam_sim import plot_results, Sol
import time
import matplotlib.animation as animation
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ckdtree

def MotionPlanningAlgorithm(realTimeDraw, map_instance):
    """
    realTimeDraw: plot the results, True or False
    """

    # Generate and draw the map
    print("START")
    print(f"{bcolors.HEADER}>> Analyse the complexity of the map{bcolors.ENDC}")
    complexity = MapGen.evaluateComplexityMap(map_instance)
    print("complexity:",complexity)
    
    if realTimeDraw:
        ax, plt, fig = plot_map(map_instance, "top") # this is the one were the primitives are plotted
        ax2, plt2, fig2 = plot_map(map_instance, "side")
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    
    # Which cost function you want to use (1-Astar 2-AdaptiveAstar 3-HeadingConstraint) 
    typeFunction = 1

    # The tuning parameter in the algorithm (if using typeFunction=3)
    dec = 0.1

    # Search the path
    print(f"{bcolors.HEADER}>> Trajectory search{bcolors.ENDC}")
    start_time = time.time()
    if not realTimeDraw:
        ax = None 
        plt = None

    # Search
    if complexity == 0:
        print(f"{bcolors.WARNING}single tree search{bcolors.ENDC}")
        trajectory, succesfulSearch, totalCost = a_star_search(ax, plt, map_instance, realTimeDraw, typeFunction, dec)
    else:
        print(f"{bcolors.WARNING}double tree search{bcolors.ENDC}")
        trajectory, succesfulSearch, totalCost = double_a_star_search(ax, plt, map_instance, realTimeDraw, typeFunction, dec)
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    end_time = time.time()

    # Get the inputs plot
    #sol = np.asarray(trajectory).T  # It should be the states on the columns
    #print(sol)
    #t_eval = np.linspace(0, 0.1*len(trajectory), len(trajectory))
    #sol = Sol(t_eval,sol)
    #plot_results(sol)
    #print("printed the inputs!")

    # Computational time (seconds)
    print(f"{bcolors.HEADER}>> A star analysis{bcolors.ENDC}")
    print(f"total time for Astar:...{end_time-start_time:.4f} seconds")
    
    # Number of vertices in the trajectory
    print(f"length of trajectory: {len(trajectory):.2f} vertices>>")
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    # Save the trajectory into saved_trajectory.csv file
    print(f"{bcolors.HEADER}>> Save the trajectory >> saved_trajectory.csv{bcolors.ENDC}")
    #df = pd.DataFrame(trajectory, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
    #df.to_csv("saved_trajectory.csv", index=False)
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

        if realTimeDraw:
            # Draw torpedo in the two created plots
            norm_index = (ind / len(trajectory)) 
            #norm_index = 0.7
            draw_torpedo(ax, vertex, norm_index)
            draw_torpedo(ax2, vertex, norm_index)

            '''
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
        '''
            
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
        step = 4  # how many points to skip
        frame_indices = range(0, len(trajectory), step)
        ani = animation.FuncAnimation(fig, update, frames=frame_indices, fargs=(ax, plt, trajectory), interval=100)

        # Save as GIF
        ani.save('/home/parallels/Desktop/smarc_modelling-master/pics/torpedo_motion.gif', writer="ffmpeg", fps=5)

        # Show animation
        print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}THE END{bcolors.ENDC}")

def MotionPlanningROS(start_state, goal_state, map_boundaries, map_resolution):
    """
    This is the function called by the ROS node. It takes:
    -)  start_state (np.array)
    -)  goal_state (np.array)
    -)  map_boundaries ((max_x, max_y, max_z))
    -)  map_resolution (float)

    And the output is:
    -)  a list of waypoints ans a successful Flag
    """

    print(">> Creating the map")
    map_instance = MapGen.generateMapInstance(start_state, goal_state, map_boundaries, map_resolution)
    print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
    
    complexity = MapGen.evaluateComplexityMap(map_instance)
    # complexity = 0
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

'''
if __name__ == "__main__":

    # # SAM initial state 
    # eta0 = np.zeros(13)
    # eta0[0] = 3#3.35140089e+00
    # eta0[1] = 0#1.39057753e-01
    # eta0[2] = 0.6#2.20124902e+00
    # eta0[3] = 0.93969262#-4.48740911e-02
    # eta0[4] = 0#-6.28489231e-01
    # eta0[5] = 0#3.44557419e-02
    # eta0[6] = -0.34202014#.75757954e-01
    # eta0[7] = 0#7.02536865e-02
    # eta0[8] = 0#1.44681268e-02
    # eta0[9] = 0#-2.48074309e-02
    # eta0[10] = 0#3.36878054e-01
    # eta0[11] = 0#1.18841045e-01
    # eta0[12] = 0#-2.79456531e-02
    # #nu0 = np.zeros(6)   # Zero initial velocities
    # u0 = np.zeros(6)    #The initial control inputs for SAM
    # u0[0] = 50          #Vbs
    # u0[1] = 50          #lcg
    # x0 = np.concatenate([eta0, u0])
    
    # # SAM final state
    # finalState = x0.copy()
    # finalState[0] = 5#5.25401611
    # finalState[1] = 1#-0.07726609
    # finalState[2] = 1.8#0.80653607
    # finalState[3] = 0.98480775#0.4092956
    # finalState[4] = 0#0.69630861
    # finalState[5] = 0.17364818#0.14047186
    # finalState[6] = 0#0.57262472


    # SAM initial state 
    eta0 = np.zeros(7)
    eta0[0] = 1.55
    eta0[1] = 0.25
    eta0[2] = 0.75
    initial_yaw = np.deg2rad(0)   # in deg
    initial_pitch = np.deg2rad(0) # in deg
    initial_roll = np.deg2rad(0)  # in deg 
    r = R.from_euler('zyx', [initial_yaw, initial_pitch, initial_roll])
    q0 = r.as_quat()
    eta0[3] = q0[3]
    eta0[4:7] = q0[0:3]
    nu0 = np.zeros(6)   # Zero initial velocities
    u0 = np.zeros(6)    #The initial control inputs for SAM
    u0[0] = 50          #Vbs
    u0[1] = 50          #lcg
    x0 = np.concatenate([eta0, nu0, u0])
    
    # SAM final state
    finalState = x0.copy()
    finalState[0:3] = (7.75, -0.25, 2.75)
    final_yaw = np.deg2rad(0)   # in deg
    final_pitch = np.deg2rad(0) # in deg
    final_roll = np.deg2rad(0)  # in deg 
    r = R.from_euler('zyx', [final_yaw, final_pitch, final_roll])
    q = r.as_quat()
    finalState[3] = q[3]
    finalState[4:7] = q[0:3]
    
    # Define the map
    map_bounds = (10, 2.5, 3, 0, -2.5, -0.5)   # (x_max, y_max, z_max, x_min, y_min, z_min)
    map_res = 0.5   # Resolution of the map_grid (used for goal area)

    # Generate a map instance compatible with the algorithm 
    map_instance = MapGen.generateMapInstance(x0, finalState, map_bounds, map_res)
    initial_and_final = [x0]
    initial_and_final.append(finalState)
    draw_map_and_toredo(map_instance, initial_and_final)

    # Test with images and GIF
    MotionPlanningAlgorithm(True, map_instance)

    #Test for ROS and draw the final path
    #trajectory, successfulFlag = MotionPlanningROS(x0, finalState, map_bounds, map_res)
    #draw_map_and_toredo(map_instance, trajectory)
'''
