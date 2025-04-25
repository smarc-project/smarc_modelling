import numpy as np
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM import SAM
#from smarc_modelling.MotionPrimitives.MapGeneration_MotionPrimitives import map_instance, TILESIZE
import smarc_modelling.motion_planning.MotionPrimitives.MapGeneration as MapGen
from smarc_modelling.motion_planning.MotionPrimitives.GenerationTree import a_star_search, double_a_star_search, body_to_global_velocity
from smarc_modelling.motion_planning.MotionPrimitives.ObstacleChecker import arrived
from smarc_modelling.motion_planning.MotionPrimitives.PlotResults import plot_map, draw_torpedo, update
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import time 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.stats import norm
from smarc_modelling.motion_planning.MotionPrimitives.trm_colors import *
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

def runStatisticalAnalysis(numberTrials, chosenComplexity):

    # Initialize variables
    dec = 0.5
    typeFunction = 3
    all_results = []
    all_trajectories = []
    all_maps = []
    wasSuccessful_list = []

    # Start analysis
    print(f"{bcolors.HEADER}>> Analysis started{bcolors.ENDC}")
    for trial in range(numberTrials):

        # Generate the map
        complexity = -1
        while complexity != chosenComplexity:
            map_instance = MapGen.generationFirstMap()

            # Analyse its complexity
            complexity = MapGen.evaluateComplexityMap(map_instance)
        print("complexity:",complexity) # 0-single, 1-xy, 2-z, 3-xyz

        # Search the trajectory
        if complexity == 0:
            typeTree = 1    # single
            print(f"{bcolors.WARNING}single tree search{bcolors.ENDC}")
            start_a_star = time.time()
            trajectory, successfulSearch, failingNotes = a_star_search(None, None, map_instance, False, typeFunction, dec)  
            end_a_star = time.time()
        else:
            typeTree = 2    # double
            print(f"{bcolors.WARNING}double tree search{bcolors.ENDC}")
            start_a_star = time.time()
            trajectory, successfulSearch, failingNotes = double_a_star_search(None, None, map_instance, False, typeFunction, dec)
            end_a_star = time.time()
        
        # Analyse the results
        if successfulSearch:
            last_state = trajectory[-1]
        else:
            last_state = map_instance["initial_state"]
        all_results.append((typeTree, complexity, successfulSearch, failingNotes, map_instance["initial_state"], map_instance["final_state"], last_state, start_a_star-end_a_star, len(trajectory)))

        # Save the trajectory
        trajectory_to_save = []
        for i in range(0, len(trajectory), 5):
            trajectory_to_save.append(trajectory[i])
        all_trajectories.append(trajectory_to_save)

        # Save successfulSearch
        wasSuccessful_list.append(successfulSearch)

        # Save the map
        all_maps.append(map_instance)

        # Store the data in the dataset
        df = pd.DataFrame(all_results, columns=["typeTree", "complexity", "success", "failingNotes", "desired_initial_state", "desired_final_state", "actual_final_state", "time_seconds", "number_vertices"])
        df.to_csv("a_star_results.csv", index=False)
        print(f"{bcolors.OKGREEN}Results are stored in a_star_results.csv file!{bcolors.ENDC}")

        # Save the trajectory
        print(f"{bcolors.HEADER}>> Save the trajectory >> saved_trajectory.csv{bcolors.ENDC}")
        df = pd.DataFrame(trajectory, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
        df.to_csv("saved_trajectory.csv", index=False)
        print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")

        
        # Create GIF (!! It will be overwritten at each trial !!)
        print(f"{bcolors.HEADER}>> Generate GIF{bcolors.ENDC}")
        ax, plt, fig = plot_map(map_instance, "gif")
        step = 10  # how many points to skip
        frame_indices = range(0, len(trajectory), step)
        ani = animation.FuncAnimation(fig, update, frames=frame_indices, fargs=(ax, plt, trajectory), interval=100)

        # Save the GIF
        ani.save('/home/parallels/Desktop/smarc_modelling-master/pics/torpedo_motion.gif', writer="ffmpeg", fps=5)
        print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
        
    
    # Draw all the maps 
    for i, traj in enumerate(all_trajectories):
        # Plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        map_x_max = map_instance["x_max"]   #number of column-wise
        map_y_max = map_instance["y_max"]   #number of row-wise
        map_z_max = map_instance["z_max"]
        ax.set_xlabel('X / East')
        ax.set_ylabel('Y / North')
        ax.set_zlabel('-Z / Down')
        ax.set_title(f"Trajectory {i+1}")
        ax.view_init(elev=75, azim=90)  # top view
        ax.set_xlim(0, map_x_max)
        ax.set_ylim(0, map_y_max)
        ax.set_zlim(0, map_z_max)
        ax.set_xlim(0, map_x_max)
        ax.set_ylim(0, map_y_max)
        ax.set_zlim(0, map_z_max)
        ax.set_box_aspect([map_x_max, map_y_max, map_z_max])

        # Plot start and goal points
        start_pos = all_maps[i]["start_pos"]   #(x,y,z)
        goal_pixel = all_maps[i]["goal_pixel"] #(x,y,z)
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='green', marker='o', s=100, label="Start")
        ax.scatter(goal_pixel[0], goal_pixel[1], goal_pixel[2], c='red', marker='x', s=100, label="End")

        if wasSuccessful_list[i]:
            for j, vertex in enumerate(traj):
                # Draw torpedo in the two created plots
                norm_index = 0.6 
                draw_torpedo(ax, vertex, norm_index)
    
    plt.show()