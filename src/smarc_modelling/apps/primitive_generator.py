import sys
import os
# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


import numpy as np
from smarc_modelling.motion_planning.MotionPrimitives.MotionPrimitives import SAM_PRIMITIVES
from smarc_modelling.motion_planning.MotionPrimitives.PlotResults import draw_torpedo
from joblib import Parallel, delayed
from threading import Lock
import multiprocessing
import matplotlib.pyplot as plt
import sys
import time
from scipy.spatial.transform import Rotation as R


def parallelise_generation(inputs, current_state, sim):
    '''This is the function that is parallelized'''

    # Initialize variables
    inputLen = len(inputs)

    # If number of inputs is != from number of indices there is something wrong
    if inputLen % 2 != 0:
        print(inputs)
        print("ERROR! INPUTS AND INDICES ARE NOT THE SAME LENGTH")
        sys.exit(1)

    # Get all the points within one single input primitive
    data, cost = sim.curvePrimitives_justToShow(current_state, inputs[0 : inputLen//2], inputs[inputLen//2 : inputLen]) 

    return data, cost

def draw_primitives(x0, list_of_primitives, typePlot = "rotated"):

    # Map boundaries
    map_x_max = 10 # meters
    map_y_max = 10 # meters
    map_z_max = 10 # meters

    # Generate the plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Axis labels
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlabel('-Z / Down')
    ax.set_title('Trajectory with Map')

    # Draw SAM in x0 configuration
    draw_torpedo(ax, x0, 0.9)

    # Decide which view you want
    match typePlot:
        case "top":
            ax.view_init(elev=75, azim=90)  # top view

        case "side":
            ax.view_init(elev=0, azim=0)  # right view

        case "rotated":
            ax.view_init(elev=35, azim=70) 

        case "gif":
            ax.view_init(elev=40, azim=0)

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

    # Draw the primitives
    for sequence_states in list_of_primitives:
        x_vals = sequence_states[0, :]
        y_vals = sequence_states[1, :]
        z_vals = sequence_states[2, :]
        ax.plot(x_vals, y_vals, z_vals, 'c+', linewidth=0.5)

    print("[ OK ]")
    plt.show()

def generate_primitives():
    print(">> Generating the primitives")
    start_time = time.time()
    simulator = SAM_PRIMITIVES()

    #------------------------------------------------------------------------------------------------------------------------------
    # Edit HERE your initial state x0
    eta0 = np.zeros(7)
    eta0[0] = 3
    eta0[1] = 3
    eta0[2] = 1.5
    eta0[3] = 1.0       # Initial quaternion (no rotation) 
    nu0 = np.zeros(6)   # Zero initial velocities
    u0 = np.zeros(6)    # The initial control inputs for SAM
    u0[0] = 50  #Vbs
    u0[1] = 50  #lcg
    x0 = np.concatenate([eta0, nu0, u0])

    # Edit HERE the inputs you want to apply
    nInputs = 5 # Vbs, lcg, ds, dr, RPM1 (=RPM2)
    step_actuator_discretisation = 2 # Discretisation for actuator angles --> np.arange(-7,7, step)
    step_vbs_discretisation = 20 # Percentage %
    step_lcg_discretisation = 20 # Percentage %
    step_rpm_discretisation = 200 # RPM
    length_of_primitive = 2 # Seconds
    #------------------------------------------------------------------------------------------------------------------------------

    # Change the length of the primitive
    simulator.changeLengthOfPrimitive(length_of_primitive)

    # Build up the input vectors
    rudder_inputs = np.arange(-6, 6, step_actuator_discretisation)
    stern_inputs = np.arange(-6, 6, step_actuator_discretisation)
    vbs_inputs = np.arange(10, 90, step_vbs_discretisation)
    lcg_inputs = np.arange(10, 90, step_lcg_discretisation)
    rpm_inputs = np.arange(-1300, 1300, step_rpm_discretisation)
    input_pairs = np.array(np.meshgrid(rudder_inputs, rpm_inputs, stern_inputs, vbs_inputs, lcg_inputs)).T.reshape(-1,nInputs)
    
    # Add the input indices
    additional_values = np.tile([3, 4, 2, 0, 1], (input_pairs.shape[0], 1))

    # Final input vector
    full_input_pairs = np.hstack((input_pairs, additional_values))

    # Parallelise the primitive generation
    
    results = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(parallelise_generation)(inputs, x0, simulator) for inputs in full_input_pairs
    ) 

    # Save the results in a list of np.array
    list_primitives = []
    for data, cost in results:
        list_primitives.append(data)

    end_time = time.time()
    print("[ OK ]   <<  seconds:", end_time-start_time)

    # Draw the primitives and SAM
    print(">> Drawing the primitives")
    draw_primitives(x0, list_primitives, "rotated")

    return list_primitives

generate_primitives()