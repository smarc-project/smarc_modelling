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

def runStatisticalAnalysis(numberTrials):

    # Initialize variables
    dec = 0.5
    typeFunction = 3
    all_results = []

    # Start analysis
    print(f"{bcolors.HEADER}>> Analysis started{bcolors.ENDC}")
    for trial in range(numberTrials):

        # Generate the map
        chosenComplexity = 0
        complexity = -1
        while complexity != chosenComplexity:
            map_instance = MapGen.generationFirstMap()

            # Analyse its complexity
            complexity_full = MapGen.evaluateComplexityMap(map_instance)
            complexity = complexity_full[0] + complexity_full[1] + complexity_full[2]
        print("complexity:",complexity) # 0-basic, 1-easy, 2-medium, 3-hard

        # Search the trajectory
        ax = None
        plt = None
        if complexity == 0:
            typeTree = 1    # single
            print(f"{bcolors.WARNING}single tree search{bcolors.ENDC}")
            trajectory, successfulSearch, failingNotes = a_star_search(ax, plt, map_instance, False, typeFunction, dec)
        else:
            typeTree = 2    # double
            print(f"{bcolors.WARNING}double tree search{bcolors.ENDC}")
            trajectory, successfulSearch, failingNotes = double_a_star_search(ax, plt, map_instance, False, typeFunction, dec)
        
        # Analyse the results
        all_results.append((typeTree, complexity, successfulSearch, failingNotes, map_instance["initial_state"], map_instance["final_state"]))

        # Store the data in the dataset
        df = pd.DataFrame(all_results, columns=["typeTree", "complexity", "success", "failingNotes", "initial_state", "final_state"])
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
