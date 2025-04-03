import smarc_modelling.motion_planning.MotionPrimitives.MapGeneration as MapGen
import numpy as np
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
import smarc_modelling.motion_planning.MotionPrimitives.MapGeneration as MapGen
from smarc_modelling.motion_planning.MotionPrimitives.GenerationTree import a_star_search, body_to_global_velocity, testOptimization
from smarc_modelling.motion_planning.MotionPrimitives.PlotResults import *
import time
import matplotlib.animation as animation
import pandas as pd

def MotionPlanningAlgorithm(realTimeDraw):
    """
    realTimeDraw: plot the results, True or False
    """

    # Generate and draw the map
    print("START")
    print(">> Generate the map")
    map_instance = MapGen.generationFirstMap()
    if realTimeDraw:
        ax, plt, fig = plot_map(map_instance, "top") # this is the one were the primitives are plotted
        ax2, plt2, fig2 = plot_map(map_instance, "side")
    print("[ OK ]")

    #plt.show()
    #exit(1)
    
    # Which cost function you want to use (1-Astar 2-AdaptiveAstar 3-HeadingConstraint) 
    typeFunction = 3

    # The tuning parameter in the algorithm (if using typeFunction=3)
    dec = 0.1

    # Search the path
    print(">> Trajectory search")
    start_time = time.time()
    if not realTimeDraw:
        ax = None 
        plt = None
    
    # Skip search or not
    onlyOptimization = False
    if not onlyOptimization:
        trajectory, succesfulSearch, totalCost = a_star_search(ax, plt, map_instance, realTimeDraw, typeFunction, dec)
    else:
        trajectory = testOptimization(map_instance)
    print("[ OK ]")
    end_time = time.time()

    # Computational time (seconds)
    print(">> A star analysis")
    print(f"total time for Astar:...{end_time-start_time:.4f} seconds")
    
    # Number of vertices in the trajectory
    print(f"length of trajectory: {len(trajectory):.2f} vertices>>")
    print("[ OK ]")
    # Save the trajectory into saved_trajectory.csv file
    print(">> Save the trajectory >> saved_trajectory.csv")
    df = pd.DataFrame(trajectory, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
    df.to_csv("saved_trajectory.csv", index=False)
    print("[ OK ]")

    # Draw SAM torpedo in the map 
    ind = 0
    if realTimeDraw:
        print(">> Draw SAM as a cylinder in the plot")
    for vertex in trajectory:

        # Print the velocity for each vertex in the trajectory
        q0, q1, q2, q3 = vertex[3:7]
        globalV = body_to_global_velocity((q0, q1, q2, q3), vertex[7:10])
        print(f"Velocity {ind:.0f} = {np.linalg.norm(globalV): .2f} m/s")

        if realTimeDraw:
            # Draw torpedo in the two created plots
            norm_index = (ind / len(trajectory)) 
            draw_torpedo(ax, vertex, norm_index)
            draw_torpedo(ax2, vertex, norm_index)

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

        ind += 1

    # Show the map, path and SAM 
    if realTimeDraw:
        plt.show()
        print("[ OK ]")
    
    # Create the gif
    gifOutput = False
    if gifOutput:
        print(">> Generate GIF")
        # Set up figure and 3D axis
        ax, plt, fig = plot_map(map_instance, "gif")

        # Create animation
        num_frames = len(trajectory)
        ani = animation.FuncAnimation(fig, update, frames=num_frames, fargs=(ax, plt, trajectory), interval=100)

        # Save as GIF
        ani.save("torpedo_motion.gif", writer="ffmpeg", fps=5)

        # Show animation
        print("[ OK ]")
    print("THE END")
        
if __name__ == "__main__":

    MotionPlanningAlgorithm(True)