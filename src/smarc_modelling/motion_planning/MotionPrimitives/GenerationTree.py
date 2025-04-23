# No optimization!

import heapq
import numpy as np
import sys
import random
from joblib import Parallel, delayed
from threading import Lock
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import time
import multiprocessing
import pandas as pd
import csv
from smarc_modelling.motion_planning.MotionPrimitives.MotionPrimitives import SAM_PRIMITIVES
from smarc_modelling.motion_planning.MotionPrimitives.ObstacleChecker import calculate_angle_betweenVectors, calculate_angle_goalVector, compute_A_point_forward
from smarc_modelling.motion_planning.MotionPrimitives.OptimizationAcados_doubleTree import optimization_acados_doubleTree
from smarc_modelling.motion_planning.MotionPrimitives.OptimizationAcados_singleTree import optimization_acados_singleTree
from smarc_modelling.motion_planning.MotionPrimitives.Optimizer.acados_trajectory_simulator import main
from smarc_modelling.motion_planning.MotionPrimitives.PlotResults import plot_waypoints
from smarc_modelling.motion_planning.MotionPrimitives.trm_colors import *
import smarc_modelling.motion_planning.MotionPrimitives.GlobalVariables as glbv
import matplotlib.pyplot as plt

# Global variables
lock = Lock()

# Classes
class Node:
    def __init__(self, state, cost=0):
        # Define the initial conditions of the node 
        self.state = state
        self.cost = cost 

    def __lt__(self, other):
        # Define when a node is "better" than another. Better based on the cost
        return self.cost < other.cost

    def __eq__(self, other):
        # Define when two nodes are equal. Equal if the have the same spatial coordinates (x,y,z)
        return self.state[0] == other.state[0] and self.state[1] == other.state[1] and self.state[2] == other.state[2]

    def __hash__(self):
        # Make Node hashable by its state
        return hash((self.state[0], self.state[1], self.state[2])) 

# Functions
def body_to_global_velocity(quaternion, body_velocity):
    """
    Convert body-fixed linear velocity to global frame.
    
    Parameters:
    - quaternion: [q0, q1, q2, q3] (unit quaternion representing orientation)
    - body_velocity: [vx, vy, vz] in body frame
    
    Returns:
    - global_velocity: [vX, vY, vZ] in global frame
    """
    
    # Create rotation object from quaternion
    rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # [x, y, z, w] format
    
    # Rotate body velocity to global frame
    global_velocity = rotation.apply(body_velocity)
    
    return global_velocity

def load_trajectory_from_csv(filename):
    """
    Reads trajectory data from a CSV file and returns it as a list of states.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of states, where each state is a list or tuple of the values
              from a row in the CSV.
    """
    trajectory = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row if it exists (optional)
        header = next(reader, None)

        for row in reader:
            # Convert the string values to appropriate data types (e.g., float, int)
            # Assuming your states are numerical, you'll likely need to convert.
            try:
                state = [float(value) for value in row]
                state = np.asarray(state, dtype=np.float32)
            except ValueError:
                print(f"Warning: Could not convert row to numbers: {row}. Skipping.")
                continue
            trajectory.append(state)
    return trajectory

def testOptimization(map_instance):
            
            '''MODIFY!'''

            start_opti = time.time()
            print("<starting optimization>")
            # Load trajectory
            res_list = load_trajectory_from_csv('/home/parallels/Desktop/smarc_modelling-master/data/OptimizationTrajectory.csv')
            # Interpolate the waypoints
            res_list = linear_interpolation_waypoints(res_list)
            
            result_list, status = optimization_acados_singleTree(res_list, map_instance)
            if status == 0:
                res_list = result_list
                print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
            else:
                print(f"{bcolors.FAIL}[ X ]{bcolors.ENDC}")
            
            end_opti = time.time()
            print(f"optimization time:...{end_opti-start_opti:.4f} seconds")
            return res_list

def linear_interpolation_waypoints(waypoints):
    res_list = []
    N = len(waypoints)
    for ii in range(N):
        # Add the current waypoint to the result list
        res_list.append(waypoints[ii])
        
        # If not the last waypoint, calculate the midpoint
        if ii == N-1:
            for _ in range(N): #OPTIMIZE IT!
                #midpoint = 0.5 * (np.array(waypoints[ii]) + np.array(waypoints[ii + 1]))
                midpoint = np.array(waypoints[ii])
                res_list.append(midpoint.tolist())
            
    return res_list

def reconstruct_path(current, parents_dict, resolution_dict, map_instance, ax, pltt):
    """
    This function reconstructs the path from the goal to the start. 
    It will return a list of states.
    """

    # Initialize the variables
    final_path = []
    
    while current is not None:

        # Check if we are the starting node
        if parents_dict[current] is None:
            final_path.append(current.state)
            current = None 
            continue
        
        # Get the list of vertices from resolution_dictionary
        res_list = resolution_dict[current]
        
        
        # Optimization Acados
        if len(final_path) == 0:
            start_opti = time.time()
            print("<starting optimization>")
            #print(res_list)

            # Save trajectory
            #df = pd.DataFrame(res_list, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
            #df.to_csv('/home/parallels/Desktop/smarc_modelling-master/data/OptimizationTrajectory.csv', index=False)
            print("initial length:...", len(res_list))

            # Interpolate the waypoints
            #res_list = linear_interpolation_waypoints(res_list)
            for _ in range(5):
                res_list.append(map_instance["final_state"])

            # Optimise them 
            result_list, status = optimization_acados_singleTree(res_list[(len(res_list)-5)//2 :], map_instance)    # Only optimize half of the primitive
            if status == 0:
                success = 1
                res_list = result_list
                print(f"{bcolors.OKGREEN}[ OK ]{bcolors.ENDC}")
            else:
                success = 0
                print(f"{bcolors.FAIL}[ X ]{bcolors.ENDC}")
            end_opti = time.time()
            print(f"optimization time:...{end_opti-start_opti:.4f} seconds")
            
        # Append vertices to the final list (reverted order)
        for vertex in res_list[::-1]:
            final_path.append(vertex)

        # Update current 
        current = parents_dict[current]

    # Return reversed path
    return final_path[::-1], success  # (path, successfulSearch)

def reconstruct_path_doubleTree(current, parents_dict, resolution_dict, map_instance, ax, pltt):
    """
    This function reconstructs the path from the goal to the start. 
    It will return a list of states.
    """

    # Initialize the variables
    final_path = []
    
    while current is not None:

        # Check if we are the starting node
        if parents_dict[current] is None:
            final_path.append(current.state)
            current = None 
            continue
        
        # Get the list of vertices from resolution_dictionary
        res_list = resolution_dict[current]

        # Append vertices to the final list (reverted order)
        for vertex in res_list[::-1]:
            final_path.append(vertex)

        # Update current 
        current = parents_dict[current]

    # Return reversed path
    return final_path[::-1] 

def process_input_pair(inputs, current_state, sim, map_instance, numberTree):
    '''This is the function that is parallelized'''

    # Initialize variables
    inputLen = len(inputs)

    # If number of inputs is != from number of indices there is something wrong
    if inputLen % 2 != 0:
        print(inputs)
        print("ERROR! INPUTS AND INDICES ARE NOT THE SAME LENGTH")
        sys.exit(1)
    
    # Calculate angle between goal and velocity (for dynamic primitives length)
    q0, q1, q2, q3 = current_state[3:7]
    vx, vy, vz = body_to_global_velocity((q0, q1, q2, q3), current_state[7:10])
    v_vector = np.array([vx, vy, vz])

    # Orientation 
    alpha = calculate_angle_goalVector(current_state, v_vector, map_instance, numberTree)

    # Get all the points within one single input primitive
    data, cost, inObs, arrived, finalState = sim.curvePrimitives(current_state, inputs[0 : inputLen//2], inputs[inputLen//2 : inputLen], map_instance, alpha, numberTree)
    if not inObs:
        neighbour = data[:,-1]
        cost_path = cost

        # Return (allPoints, (lastPoint, costPath), boolInObstacle, boolArrived, stateInGoal(or None))
        return data, (neighbour, cost_path), inObs, arrived  
    
    # If no valid primitive is available, return an empty array
    return np.array([]), ([], -1), inObs, arrived

def compute_current_orientationVector(state, map_inst, numberTree, type = "normal"):
    """
    Returns either forward_orientation or backward_orientation vector based on the minimum angle between 
    goal vector and backward/forward vector
    """

    # Define quaternion
    q0,q1,q2,q3 = state[3:7]

    # Define forward and backward vectors
    rotation = R.from_quat([q1, q2, q3, q0])
    body_forward = np.array([1.0, 0.0, 0.0])
    body_backward = np.array([-1.0, 0.0, 0.0])
    forward_vector = rotation.apply(body_forward)
    forward_vector_norm = np.linalg.norm(forward_vector)
    backward_vector = rotation.apply(body_backward)
    backward_vector_norm = np.linalg.norm(backward_vector)
    forward_vector /= forward_vector_norm
    backward_vector /= backward_vector_norm

    # OLD -Compute angle 
    '''
    # Velocity vector
    vx, vy, vz = body_to_global_velocity((q0, q1, q2, q3), state[7:10])
    v_vector = np.array([vx, vy, vz])

    angle = calculate_angle_betweenVectors(v_vector, forward_vector)
    if angle < np.deg2rad(90):
        orientation_vector = forward_vector # is normalized
    else:
        orientation_vector = backward_vector # is normalized
    '''

    # Compute orientation closer to goal direction
    '''
    match type:
        case "pointA":
            state = compute_A_point_forward(state)
    '''

    angle = calculate_angle_goalVector(state, forward_vector, map_inst, numberTree, type)
    if angle < np.pi/2:
        orientation_vector = forward_vector
    else:
        orientation_vector = backward_vector

    return orientation_vector

def compute_current_velocity_pointA(vertex):
    """
    It computes the global velocity vector of pointA
    """

    # Define parameters
    globalV = body_to_global_velocity(vertex[3:7], vertex[7:10])
    v_CG_inertial = np.array([globalV[0], globalV[1], globalV[2]])        # Linear velocity of CG in inertial frame
    rr, ww, vv = vertex[10:13]
    omega_body = np.array([rr, ww, vv])              # Angular velocity in body frame
    r_fwd_body = np.array([0.655, 0, 0])           # Position of forward point relative to CG in body frame

    # Ensure quaternion is in (x, y, z, w) format for scipy
    rotation = R.from_quat(vertex[3:7])
    R_b2i = rotation.as_matrix()  # Body to inertial rotation matrix

    # Compute cross product in body frame
    v_relative_body = np.cross(omega_body, r_fwd_body)

    # Rotate to inertial frame
    v_relative_inertial = R_b2i @ v_relative_body

    # Add to CG velocity
    v_fwd_inertial = v_CG_inertial + v_relative_inertial

    return v_fwd_inertial

def get_neighbors(current, sim, map_instance, numberTree):
    """
    This function is used to compute the motion primitives for the current state.

    This function will return:
    1) A list containing all the valid primitives (all the states within all the valid primitives)
    2) A list containing only the last states of the valid primitives
    3) True/False based on if at least one primitive arrived at the goal
    4) The final state and cost if we arrived at the goal
    """
    
    dynamic_step = 3

    # Initialize variables
    max_input = 7
    step_input = dynamic_step
    reached_states = []
    last_states = []

    # Change the inputs for the primitives
    '''
    The inputs are defined like: inputs = (values, indices of u)

    An example of inputs: 
    --> If I want to change only the RPM to 500, I will write: full_input_pairs = np.array([[500, 4]])
    '''

    # 1 # Define the inputs 
    rudder_inputs = np.arange(-max_input, max_input, step_input)
    stern_inputs = np.array([-7, 0, 7])
    vbs_inputs = np.array([10, 50, 90])
    lcg_inputs = np.array([0, 50, 100])
    rpm_inputs = np.arange(-1000, 1000, 200)

    # 2 # Add the name of the input into np.meshgrid(), and change the second value of .reshape(., THIS)
    input_pairs = np.array(np.meshgrid(rudder_inputs, rpm_inputs, vbs_inputs, lcg_inputs, stern_inputs)).T.reshape(-1,5)

    # 3 # Add the index in u of the input you modified in np.tile([..., HERE], ...)
    additional_values = np.tile([3, 4, 0, 1, 2], (input_pairs.shape[0], 1))

    # 4 # Do not touch
    full_input_pairs = np.hstack((input_pairs, additional_values))

    # 5 # Control all the inputs for tests if needed 
    #full_input_pairs = np.array([[-1000, 4]]) 

    # Parallelize the creation of primitives
    arrived = False
    results = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(process_input_pair)(inputs, current.state, sim, map_instance, numberTree) for inputs in full_input_pairs
    ) 

    # Save the generated primitives
    arrived_atLeast_one = False
    finalState = None
    finalCost = None
    bestFinalAngle = 2*np.pi
    for data, last_state, inObstacle, arrived in results:

        # Check for valid primitives
        if not inObstacle: 

            # Append all the points within all the primitives
            reached_states.append(data)

            # Append only the last state of the primitives
            last_states.append(last_state)

            # Did we arrive to the goal postion? pick the best orientation with the goal!
            if arrived:

                arrived_atLeast_one = True
                bestOrientation = compute_current_orientationVector(last_state[0], map_instance, numberTree)
                finalAngle = calculate_angle_goalVector(last_state[0], bestOrientation, map_instance, numberTree)
                if finalAngle < bestFinalAngle:

                    finalState = last_state[0]
                    finalCost = last_state[1]
    end_p = time.time()

    return reached_states, last_states, arrived_atLeast_one, (finalState, finalCost)

def heuristic(state, goal_p):
    """
    This function define the basic heuristic (straight line) of the A star algorithm
    """

    return np.sqrt((state[0] - goal_p[0]) ** 2 + (state[1] - goal_p[1]) ** 2 + (state[2] - goal_p[2]) ** 2)

def calculate_f(neighbor, map_instance, tentative_g, heuristic_cost, dec, typeF, numberTree):
    """
    Here you can decide how to compute the cost f.
    1) Normal A star: 
        f = g + h
    2) Adaptive A star.
        f = g + tau * h = (g^2 + h^2)/g
    3) With heading error
        f = g_normalized + h_normalized + w * heading_error_normalized
    """

    match typeF:
        case 1:
            '''# 1 # Normal A star'''

            total_f = tentative_g + heuristic_cost

        case 2:
            '''# 2 # Adaptive A star'''

            total_f = (tentative_g**2 + heuristic_cost**2) / tentative_g

        case 3:
            '''# 3 # Using goal-distance projection on current velocity vector'''

            # Define current and goal positions
            x = neighbor[0]
            y = neighbor[1]
            z = neighbor[2]
            if numberTree == 1:
                x_goal = map_instance["goal_pixel"][0]
                y_goal = map_instance["goal_pixel"][1]
                z_goal = map_instance["goal_pixel"][2]
            else:
                x_goal = map_instance["start_pos"][0]
                y_goal = map_instance["start_pos"][1]
                z_goal = map_instance["start_pos"][2]

            # Define the distance between position and goal
            dx = x_goal - x
            dy = y_goal - y
            dz = z_goal - z

            # Define current linear velocity vector
            q0, q1, q2, q3 = neighbor[3:7]
            vx, vy, vz = body_to_global_velocity((q0, q1, q2, q3), neighbor[7:10])
            v_vector = np.array([vx, vy, vz])

            # Define the goal vector
            goal_vector = np.array([dx, dy, dz])
            goal_vector_norm = np.linalg.norm(goal_vector)
            
            # Compute the best orientation vector and normalize it
            orientation_vector = compute_current_orientationVector(neighbor, map_instance, numberTree)
            orientation_vector_norm = np.linalg.norm(orientation_vector)
            if orientation_vector_norm != 0:
                orientation_vector /= np.linalg.norm(orientation_vector)

            # Find the angle between final_vector and goal
            final_vector = orientation_vector + v_vector
            angle_between_vectors = calculate_angle_goalVector(neighbor, final_vector, map_instance, numberTree)

            # Compute d and c distances
            maxAngle = np.deg2rad(7)
            if angle_between_vectors > maxAngle:
                angleBrake = angle_between_vectors - maxAngle
                d = -1**2 / (-2*dec)
                c = np.sqrt(goal_vector_norm**2 + d**2 - 2*goal_vector_norm*d*np.cos(angleBrake))
                heuristic = c + d
            else:
                heuristic = goal_vector_norm

            total_f = (tentative_g**2 + heuristic**2) / tentative_g

        case 4:

            '''# Computing Case 3'''
            
            # Define current and goal positions
            x = neighbor[0]
            y = neighbor[1]
            z = neighbor[2]
            x_goal = map_instance["goal_pixel"][0]
            y_goal = map_instance["goal_pixel"][1]
            z_goal = map_instance["goal_pixel"][2]

            # Define the distance between position and goal
            dx = x_goal - x
            dy = y_goal - y
            dz = z_goal - z

            # Define current linear velocity vector
            q0, q1, q2, q3 = neighbor[3:7]
            vx, vy, vz = body_to_global_velocity((q0, q1, q2, q3), neighbor[7:10])
            v_vector = np.array([vx, vy, vz])

            # Define the goal vector
            goal_vector = np.array([dx, dy, dz])
            goal_vector_norm = np.linalg.norm(goal_vector)
            
            # Compute the best orientation vector and normalize it
            orientation_vector = compute_current_orientationVector(neighbor, map_instance, numberTree)
            orientation_vector_norm = np.linalg.norm(orientation_vector)
            if orientation_vector_norm != 0:
                orientation_vector /= np.linalg.norm(orientation_vector)

            # Find the angle between final_vector and goal
            final_vector = orientation_vector + v_vector
            angle_between_vectors = calculate_angle_goalVector(neighbor, final_vector, map_instance, numberTree)

            # Compute d and c distances
            maxAngle = np.deg2rad(7)
            if angle_between_vectors > maxAngle:
                angleBrake = angle_between_vectors - maxAngle
                d = -1**2 / (-2*dec)
                c = np.sqrt(goal_vector_norm**2 + d**2 - 2*goal_vector_norm*d*np.cos(angleBrake))
                heuristic = c + d
            else:
                heuristic = goal_vector_norm
            
            '''Adding the difference between the two tips'''
            # Compute the goal_vector tip position
            goal_vector_tip = map_instance["goal_pixel_pointA"]
            cg_position = np.array([x, y, z])

            # Compute forward orientation vector tip position
            rotation = R.from_quat([q1, q2, q3, q0])
            body_forward = np.array([1, 0.0, 0.0])
            forward_vector = rotation.apply(body_forward)
            forward_vector_norm = np.linalg.norm(forward_vector)
            forward_vector /= forward_vector_norm
            #current_fwd_vector = cg_position + forward_vector

            # Compute current velocity of pointA
            v_pointA = compute_current_velocity_pointA(neighbor)

            # Sum them up 
            current_vector_tip = forward_vector + v_pointA + cg_position

            # Compute the difference between the tips
            difference_tips = np.linalg.norm(goal_vector_tip - current_vector_tip)

            # Add it to the heuristic
            heuristic += difference_tips

            # Total cost function
            total_f = heuristic + tentative_g

                
            '''Computing Case 3 for pointA'''
            '''
            # Velocity vector at pointA (global frame)
            v_CG_inertial = np.array([vx, vy, vz])        # Linear velocity of CG in inertial frame
            rr, ww, vv = neighbor[10:13]
            omega_body = np.array([rr, ww, vv])              # Angular velocity in body frame
            r_fwd_body = np.array([0.655, 0, 0])           # Position of forward point relative to CG in body frame
            q = np.array([q0, q1, q2, q3])                # Quaternion (x, y, z, w) or (w, x, y, z)
            rotation = R.from_quat([q1, q2, q3, q0])
            R_b2i = rotation.as_matrix()  # Body to inertial rotation matrix
            v_relative_body = np.cross(omega_body, r_fwd_body)
            v_relative_inertial = R_b2i @ v_relative_body
            v_vector = v_CG_inertial + v_relative_inertial
            

            # Define current and goal positions for pointA
            x, y, z = compute_A_point_forward(neighbor)
            x_goal = map_instance["goal_pixel_pointA"][0]
            y_goal = map_instance["goal_pixel_pointA"][1]
            z_goal = map_instance["goal_pixel_pointA"][2]

            # Define the distance between position and goal
            dx = x_goal - x
            dy = y_goal - y
            dz = z_goal - z

            # Define the goal vector
            goal_vector = np.array([dx, dy, dz])
            goal_vector_norm = np.linalg.norm(goal_vector)

            
            # Compute the best orientation vector and normalize it
            orientation_vector = compute_current_orientationVector(neighbor, map_instance,"pointA") 
            orientation_vector_norm = np.linalg.norm(orientation_vector)
            if orientation_vector_norm != 0:
                orientation_vector /= np.linalg.norm(orientation_vector)

            # Find the angle between final_vector and goal
            final_vector = orientation_vector + v_vector
            angle_between_vectors = calculate_angle_goalVector([x, y, z], final_vector, map_instance, "pointA")   

            # Compute d and c distances
            if angle_between_vectors > maxAngle:
                angleBrake = angle_between_vectors - maxAngle
                d = -1**2 / (-2*dec)
                c = np.sqrt(goal_vector_norm**2 + d**2 - 2*goal_vector_norm*d*np.cos(angleBrake))
                heuristic += c + d
            else:
                heuristic += goal_vector_norm
            
            total_f = (tentative_g**2 + heuristic**2) / tentative_g
            '''
            '''
            # Define current and goal positions
            x_cg = neighbor[0]
            y_cg = neighbor[1]
            z_cg = neighbor[2]
            pointA = compute_A_point_forward(neighbor)
            x_goal_cg = map_instance["goal_pixel"][0]
            y_goal_cg = map_instance["goal_pixel"][1]
            z_goal_cg = map_instance["goal_pixel"][2]
            x_goal_pointA = map_instance["goal_pixel_pointA"][0]
            y_goal_pointA = map_instance["goal_pixel_pointA"][1]
            z_goal_pointA = map_instance["goal_pixel_pointA"][2]

            # Define the distance between position and goal
            dx_cg = x_goal_cg - x_cg
            dy_cg = y_goal_cg - y_cg
            dz_cg = z_goal_cg - z_cg
            dx_pointA = x_goal_pointA - pointA[0]
            dy_pointA = y_goal_pointA - pointA[1]
            dz_pointA = z_goal_pointA - pointA[2]

            # Define the goal vectors
            goal_vector_cg = np.array([dx_cg, dy_cg, dz_cg])
            goal_vector_cg_norm = np.linalg.norm(goal_vector_cg)    #a
            goal_vector_pointA = np.array([dx_pointA, dy_pointA, dz_pointA])
            goal_vector_pointA_norm = np.linalg.norm(goal_vector_pointA)    #d

            # Define the ramaining vectors for computing the areas
            dx_b = x_goal_pointA - x_cg
            dy_b = y_goal_pointA - y_cg
            dz_b = z_goal_pointA - z_cg
            b_norm = np.linalg.norm([dx_b, dy_b, dz_b]) #b
            c_norm = 0.655   #c  (half length of SAM)
            e_norm = 0.655   #e  (half length of SAM)

            # Computing the area using Heron's formula
            S1 = 0.25 * np.sqrt(4 * goal_vector_cg_norm**2 * b_norm**2 - (goal_vector_cg_norm**2 + b_norm**2 - c_norm**2)**2)
            S2 = 0.25 * np.sqrt(4 * b_norm**2 * e_norm**2 - (b_norm**2 + e_norm**2 - goal_vector_pointA_norm**2)**2)
            S = S1 + S2

            # Heuristic 
            heuristic = 2*S + goal_vector_pointA_norm

            # Plus case 3
            # Define current linear velocity vector
            q0, q1, q2, q3 = neighbor[3:7]
            vx, vy, vz = body_to_global_velocity((q0, q1, q2, q3), neighbor[7:10])
            v_vector = np.array([vx, vy, vz])
            
            # Compute the best orientation vector and normalize it
            orientation_vector = compute_current_orientationVector(neighbor, map_instance)
            orientation_vector_norm = np.linalg.norm(orientation_vector)
            if orientation_vector_norm != 0:
                orientation_vector /= np.linalg.norm(orientation_vector)

            # Find the angle between final_vector and goal
            final_vector = orientation_vector + v_vector
            angle_between_vectors = calculate_angle_goalVector(neighbor, final_vector, map_instance)

            # Compute d and c distances
            maxAngle = np.deg2rad(7)
            if angle_between_vectors > maxAngle:
                angleBrake = angle_between_vectors - maxAngle
                d = -1**2 / (-2*dec)
                c = np.sqrt(goal_vector_cg_norm**2 + d**2 - 2*goal_vector_cg_norm*d*np.cos(angleBrake))
                heuristic += c + d
            else:
                heuristic += goal_vector_cg_norm

            
            total_f = (tentative_g**2 + heuristic**2) / tentative_g
            '''

        case 5:
            # Define current and goal positions
            x = neighbor[0]
            y = neighbor[1]
            z = neighbor[2]
            x_goal = map_instance["goal_pixel"][0]
            y_goal = map_instance["goal_pixel"][1]
            z_goal = map_instance["goal_pixel"][2]
            finalRoll = np.rad2deg(map_instance["goal_pixel"][0])
            finalPitch = np.rad2deg(map_instance["goal_pixel"][1])
            finalYaw = np.rad2deg(map_instance["goal_pixel"][2])

            # Define the distance between position and goal
            dx = x_goal - x
            dy = y_goal - y
            dz = z_goal - z

            # Define current linear velocity vector
            q0, q1, q2, q3 = neighbor[3:7]

            # Define the goal vector
            goal_vector = np.array([dx, dy, dz])
            goal_vector_norm = np.linalg.norm(goal_vector)

            d = goal_vector_norm
            r = R.from_quat([q1, q2, q3, q0])
            roll, pitch, yaw = r.as_euler('xyz', degrees=True)

            heuristic = np.sqrt(d**2 + (pitch - finalPitch)**2 + (yaw - finalYaw)**2)
            #total_f = (tentative_g**2 + heuristic**2) / tentative_g
            total_f = tentative_g + heuristic

        case _:
            print("Non valid cost function f...")
            total_f = 0
        
    return total_f

def getResolution(reached_states, dt_reference):
    # Change to global
    dt_primitives = glbv.DT_PRIMITIVES
    dt_current = 0
    list_vertices = []

    if dt_reference < dt_primitives:
        print("!! The reference dt for resolution is less than the dt used for generating the primitives !!")
        return list_vertices.append(reached_states[:, -1])
    
    # Get resolution
    for index in range(reached_states.shape[1]):
        vertex = reached_states[:, index]
        if not np.array_equal(vertex, reached_states[:, 0]) and not np.array_equal(vertex, reached_states[:, -1]):
            if dt_current >= dt_reference:
                list_vertices.append(vertex)
                dt_current = 0

        dt_current += dt_primitives
    list_vertices.append(reached_states[:, -1])

    return list_vertices

def computeAngleDeg(state1, state2):
        r1 = R.from_quat(state1[3:7])
        r2 = R.from_quat(state2[3:7])
        q_rel = r2 * r1.inv()
        angle_rad = q_rel.magnitude()

        return np.rad2deg(angle_rad)

def find_tree_intersection(g_cost_tree1, g_cost_tree2, list_connection_states, minimumDistance=0.5, minimumAngle = 20):
    connection_node1 = None
    connection_node2 = None
    foundConnection = False
    coords_tree1 = np.array([node.state[:3] for node in g_cost_tree1])  # x, y, z only
    coords_tree2 = np.array([node.state[:3] for node in g_cost_tree2])

    tree1_kdtree = KDTree(coords_tree1)
    for idx2, coord2 in enumerate(coords_tree2):

        distance, index = tree1_kdtree.query(coord2)
        if distance < minimumDistance:
            node1 = list(g_cost_tree1.keys())[index]
            node2 = list(g_cost_tree2.keys())[idx2]
            angle_deg = computeAngleDeg(node1.state, node2.state)

            if np.abs(angle_deg) < minimumAngle:
                list_connection_states.append((node1.state, node2.state))
                #list_connection_states.append(node1.state)
                #list_connection_states.append(node2.state)

            #print(f"Intersection found between Tree1 and Tree2: {node1.state}, {node2.state}")
            #return node1, node2  # Return the connecting nodes

    '''
    # Run MPC if necessary
    if len(list_connection_states) != 0:
        #print("Found possible connections")
        foundConnection = True
        # Take the one which have a similar pitch
        connection_node1, connection_node2 = findBestConnectionNodes(list_connection_states)
        #print(connection_node1)
        #print(connection_node2)
    '''

    '''
    list_connection_states = []  #[(node1, node2), ...]
    tree1_kdtree = KDTree(coords_tree1)
    for idx2, coord2 in enumerate(coords_tree2):
        
        # Compute distance
        distance, index = tree1_kdtree.query(coord2)

        # Consider the two nodes
        state_node1 = list(g_cost_tree1.keys())[index].state
        state_node2 = list(g_cost_tree2.keys())[idx2].state

        # Compute their orientation
        r1 = R.from_quat(state_node1[3:7])
        r2 = R.from_quat(state_node2[3:7])
        q_rel = r2 * r1.inv()
        angle_deg = np.rad2deg(q_rel.magnitude())

        # Compute the cost of the connection
        if angle_deg < 20 and distance < 2:
            list_connection_states.append((state_node1, state_node2, distance))
            #print(f"Intersection found between Tree1 and Tree2: {node1.state}, {node2.state}")
            #return node1, node2  # Return the connecting nodes

    # Run MPC if necessary
    if len(list_connection_states) != 0:
        #print("Found possible connections")
        foundConnection = True
        # Take the one which have a similar pitch
        connection_node1, connection_node2 = findBestConnectionNodes(list_connection_states)
        #print(connection_node1)
        #print(connection_node2)
    #print("No intersections!")
    '''

    moreThanMinimum = False
    if len(list_connection_states) > 100:
        moreThanMinimum = True
    
    return (list_connection_states, moreThanMinimum)

def findBestConnectionNodes(list_states):
    best_angle = np.inf
    best_dist = np.inf

    # Using forward vector
    '''
    for state1, state2 in list_states:
        forward_1 = compute_current_forward_vector(state1)  # in deg
        forward_2 = compute_current_forward_vector(state2)  # in deg
        angle_between_vectors = calculate_angle_betweenVectors(forward_1, forward_2)
        if angle_between_vectors < best_angle:
            best_angle = angle_between_vectors
            best_state1 = state1
            best_state2 = state2
    '''
    '''
    # Using difference between quaternions
    for state1, state2 in list_states:
        r1 = R.from_quat(state1[3:7])
        r2 = R.from_quat(state2[3:7])
        q_rel = r2 * r1.inv()
        angle_rad = q_rel.magnitude()
        if angle_rad < best_angle:
            best_angle = angle_rad
            best_state1 = state1
            best_state2 = state2
    '''

    # Using current_v + forward angles
    for state1, state2 in list_states:
        #f1 = find_f_vector(state1)
        #f2 = find_f_vector(state2)
        v1 = body_to_global_velocity(state1[3:7], state1[7:10])
        v2 = - body_to_global_velocity(state2[3:7], state2[7:10])
        angle_deg = np.rad2deg(calculate_angle_betweenVectors(v1, v2))
        if angle_deg < best_angle:
            best_angle = angle_deg
            best_state1 = state1
            best_state2 = state2
    
    # Delete the best state from the list 
    for i, (state1, state2) in enumerate(list_states):
        if np.array_equal(state1, best_state1) and np.array_equal(state2, best_state2):
            del list_states[i]
            break
    
    # Create the list containing the best states
    connection_list = [best_state1]
    connection_list.append(best_state2)
    
    return connection_list

    #Using the distance
    '''
    for state1, state2, distance in list_states:
        dist = distance
        if dist < best_dist:
            best_dist = dist
            best_state1 = state1
            best_state2 = state2
    '''

    #return best_state1, best_state2

def find_f_vector(state):

    """The f vector is the sum between the current v_global and the current forward orientation"""

    current_v = body_to_global_velocity(state[3:7], state[7:10])
    current_forward = compute_current_forward_vector(state)

    return current_v + current_forward

def compute_current_pitch(state):

    q = state[3:7]
    r = R.from_quat(q)
    _, pitch, yaw = r.as_euler('xyz', degrees=True)

    return pitch # in deg 

def compute_current_forward_vector(state):

    # Define quaternion
    q0,q1,q2,q3 = state[3:7]

    # Define forward and backward vectors
    rotation = R.from_quat([q1, q2, q3, q0])
    body_forward = np.array([1.0, 0.0, 0.0])
    forward_vector = rotation.apply(body_forward)
    forward_vector_norm = np.linalg.norm(forward_vector)
    forward_vector /= forward_vector_norm

    return forward_vector

def double_a_star_search(ax, plt, map_instance, realTimeDraw, typeF_function, dec):
    """
    This is the main function of the algorithm. This function runs the main loop for generating the path.
    If needed, change the initial condition of SAM in the "SAM initial state".
    If you want to change the starting position (or goal position), use MapGeneration_MotionPrimitives.py.

    This function returns (trajectory, successfulZeroOrOne, totalCost)
    """

    # Initialise general variables (valid for both trees)
    random.seed()
    sim = SAM_PRIMITIVES()
    dt_resolution = glbv.RESOLUTION_DT
    flag = 0    # for number of iterations
    nMaxIterations = 300

    # First tree variables
    x0 = map_instance["initial_state"]
    start = Node(x0)
    open_set = []                   # (cost_node, Node)
    parents_dictionary = {}         # (Node_current: Node_parent)
    g_cost = {start: 0}             # keeps track of the costs of the nodes, useful to understand more convenient paths
    resolution_dictionary = {}
    heapq.heappush(open_set, (0, start))
    parents_dictionary[start] = None
    arrivedPoint = False

    # Second tree variables
    x0_secondTree = map_instance["final_state"] 
    start_secondTree = Node(x0_secondTree)
    open_set_secondTree = []
    parents_dictionary_secondTree = {}
    g_cost_secondTree = {start_secondTree: 0}
    resolution_dictionary_secondTree = {}
    heapq.heappush(open_set_secondTree, (0, start_secondTree))
    parents_dictionary_secondTree[start_secondTree] = None
    arrivedPoint_secondTree = False
    

    # Start the search
    list_connection_states = []  #[(node1, node2), ...]
    while (flag < nMaxIterations):   ###Change this in the future
        # Are there intersections?
        if flag > 0:
            #connection_node1, connection_node2, found_connection = find_tree_intersection(g_cost, g_cost_secondTree, list_connection_states)
            status = 1

            list_connection_states, moreThanMinimum = find_tree_intersection(g_cost, g_cost_secondTree, list_connection_states, 1, 50)
            if arrivedPoint and arrivedPoint_secondTree and not moreThanMinimum:
                
                distance = 0
                while len(list_connection_states) == 0 and distance <= 11:
                    distance += 1
                    angle = 10
                    print("trying distance=", distance)
                    while angle <= 180:
                        angle += 10
                        list_connection_states, moreThanMinimum = find_tree_intersection(g_cost, g_cost_secondTree, list_connection_states, distance, angle)
                print("Angle (deg):", angle)
                moreThanMinimum = True

            currentNumOptimization = 0
            if moreThanMinimum: 
                print(f"{bcolors.OKBLUE}Double Astar is starting connecting the trees{bcolors.ENDC}")
                while(status!=0):

                    maxNumOptimizations = 5  
                    if currentNumOptimization > maxNumOptimizations:
                        print(f"{bcolors.FAIL}EXCEEDED MAX NUMBER OF OPTIMIZATIONS! - exit {bcolors.ENDC}")
                        return list_connection_states, 0, "maxNumberOptimizations"
                    
                    if len(list_connection_states) == 0:
                        print(f"{bcolors.FAIL}IMPOSSIBLE TO CONNECT THE TREES! - exit {bcolors.ENDC}")
                        return list_connection_states, 0, "NoPointsToConnect"

                    currentNumOptimization += 1
                    list_connection = findBestConnectionNodes(list_connection_states)  

                    # Reconstruct the second path and invert v, w, rpm
                    waypoints = []
                    first_path = reconstruct_path_doubleTree(Node(list_connection[0]), parents_dictionary,resolution_dictionary, map_instance, ax, plt)
                    second_path = reconstruct_path_doubleTree(Node(list_connection[-1]), parents_dictionary_secondTree, resolution_dictionary_secondTree, map_instance, ax, plt)

                    # Add last states for robustness
                    for _ in range(50):
                        second_path.insert(0, second_path[0])
                    
                    for waypoint in second_path:
                        reverted_waypoint = waypoint.copy()
                        reverted_waypoint[7:10] = -reverted_waypoint[7:10]
                        reverted_waypoint[17:] = -reverted_waypoint[17:]
                        reverted_waypoint[10:13] = -reverted_waypoint[10:13]
                        waypoints.append(reverted_waypoint)

                    # Create the list containing the two nodes to be connected
                    list_connection_full = []
                    for _ in range(10):
                        list_connection_full.append(list_connection[0])
                    list_connection_full.append(second_path[-1])
                    
                    # Optimizing the connection
                    print(f"{bcolors.OKBLUE}Optimizing path for connection{bcolors.ENDC}")
                    #df = pd.DataFrame(list_connection_full, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
                    #df.to_csv("list_connection_full.csv", index=False)
                    #list_connection_full = load_trajectory_from_csv('/home/parallels/Desktop/smarc_modelling-master/list_connection_full.csv')

                    connection_list_optimized, status = optimization_acados_doubleTree(list_connection_full, map_instance)
                    
                    if status != 0:
                        print(f"{bcolors.FAIL}OPTIMIZATION FAILED! trying new connection points!{bcolors.ENDC}")
                        continue
                    
                    print(f"{bcolors.OKGREEN}OPTIMIZATION SUCCEEDED!{bcolors.ENDC}")

                    # Define Q for reverting the second path
                    Q_diag = np.ones(19)
                    Q_diag[ 0:3 ] = 15e1         # Position
                    Q_diag[ 3:7 ] = 15e2         # Quaternion
                    Q_diag[ 7:10] = 13e1        # linear velocity
                    Q_diag[10:13] = 10e1         # Angular velocity
                    Q_diag[13:15] = 0        # VBS, LCG
                    Q_diag[15:17] = 0        # stern_angle, rudder_angle
                    Q_diag[17:  ] = 0        # RPM1 And RPM2
                    Q_diag[13:  ] = Q_diag[13:  ]
                    Q = np.diag(Q_diag)

                    # Change the last vertex of the second path with the newly found from the optimization of the connection
                    waypoints[-1] = connection_list_optimized[-1]
                    

                    # Optimize the second path
                    
                    array_waypoints = np.asarray(waypoints[::-1])
                    N_hor = array_waypoints.shape[0] // 2
                    T_s = 0.1
                    optimized_waypoints, status = main(array_waypoints, Q, N_hor, T_s)
                    if status != 0:
                        print(f"{bcolors.FAIL}Optimization failed - change connection points{bcolors.ENDC}")
                        continue
        

                    # Create the entire path
                    full_path_waypoints = []
                    for ii in range(len(first_path) - 1):
                        full_path_waypoints.append(first_path[ii])
                    full_path_waypoints = full_path_waypoints + connection_list_optimized
                    for ii in np.arange(1, len(optimized_waypoints)):
                        full_path_waypoints.append(optimized_waypoints[ii])


                    #optimized_waypoints = [connection_list[0]] + optimized_waypoints
                    return full_path_waypoints, 1, "success"
                
                # Optimization failed
                print(f"{bcolors.FAIL}all the optimizations failed - exit{bcolors.ENDC}")
                return [], 0, "allOptimizationsFailed"
        
        # Print the iteration number
        flag = flag + 1
        print(f"iteration {flag:.0f}")

        if arrivedPoint:
            print(f"{bcolors.OKCYAN}first arrived{bcolors.ENDC}")
        if arrivedPoint_secondTree:
            print(f"{bcolors.OKCYAN}second arrived{bcolors.ENDC}")

        # Get the current node for the first tree (the one with cheapest f_cost in open_set)
        if not arrivedPoint:
            _, current_node = heapq.heappop(open_set)   #removes and returns the node with lowest f value
            current_g = g_cost[current_node]

        # Get the current node for the second tree (the one with cheapest f_cost in open_set)
        if not arrivedPoint_secondTree:
            _, current_node_secondTree = heapq.heappop(open_set_secondTree)   #removes and returns the node with lowest f value
            current_g_secondTree = g_cost_secondTree[current_node_secondTree]
        

        # Stop the algorithm if we exceed the maximum number of iterations
        if flag > nMaxIterations:
            break

        # Find new neighbors (last point of the primitives) using the motion primitives
        if not arrivedPoint:
            reached_states, last_states, arrivedPoint, final = get_neighbors(current_node, sim, map_instance, 1)
            finalLast = final[0] # in case we arrived
            finalCost = final[1] # in case we arrived 

        # Find new neighbors for second tree (last point of the primitives) using the motion primitives
        if not arrivedPoint_secondTree:
            reached_states_secondTree, last_states_secondTree, arrivedPoint_secondTree, final_secondTree = get_neighbors(current_node_secondTree, sim, map_instance, 2)
            finalLast_secondTree = final_secondTree[0] # in case we arrived
            finalCost_secondTree = final_secondTree[1] # in case we arrived 

        # If all the generated primitives are not in the free space, then continue with the next vertex
        #if len(reached_states) == 0: 
        #    continue
        
        # Analyze the single steps within the primitives (each dt) for first tree
        if not arrivedPoint:
            if len(reached_states) != 0:
                for sequence_states in reached_states:

                    # Save specific points for resolution of the trajectory
                    list_vertices = getResolution(sequence_states, dt_resolution)
                    resolution_dictionary[Node(sequence_states[:,-1])] = list_vertices

                    # Plot the found motion primitives
                    if realTimeDraw:
                        x_vals = sequence_states[0, :]
                        y_vals = sequence_states[1, :]
                        z_vals = sequence_states[2, :]
                        ax.plot(x_vals, y_vals, z_vals, 'c+', linewidth=0.5)

                if realTimeDraw:
                    plt.draw()
                    plt.pause(0.01)

                # Save the new valid primitives
                for neighbor, cost_path in last_states:
                    
                    # Calculate tentative g score
                    tentative_g_cost = current_g + cost_path
                    
                    # Update the hierarchy
                    if Node(neighbor) not in g_cost or tentative_g_cost < g_cost[Node(neighbor)]:

                        # Save g cost in the g_cost dictionary
                        g_cost[Node(neighbor)] = tentative_g_cost              
                        
                        # Compute the f_cost
                        f_cost = calculate_f(neighbor, map_instance, tentative_g_cost,  heuristic(neighbor, (map_instance["goal_pixel"][0], map_instance["goal_pixel"][1], map_instance["goal_pixel"][2])), dec, typeF_function, 1)

                        # Add node dependency on current node
                        parents_dictionary[Node(neighbor)] = (current_node)   

                        # Save the last node of the primitive along its f_cost
                        heapq.heappush(open_set, (f_cost, Node(neighbor)))   

        # Analyze the single steps within the primitives (each dt) for second tree
        if not arrivedPoint_secondTree:
            if len(reached_states_secondTree) != 0:
                for sequence_states_secondTree in reached_states_secondTree:

                    # Save specific points for resolution of the trajectory
                    list_vertices_secondTree = getResolution(sequence_states_secondTree, dt_resolution)
                    resolution_dictionary_secondTree[Node(sequence_states_secondTree[:,-1])] = list_vertices_secondTree

                    # Plot the found motion primitives
                    if realTimeDraw:
                        x_vals_secondTree = sequence_states_secondTree[0, :]
                        y_vals_secondTree = sequence_states_secondTree[1, :]
                        z_vals_secondTree = sequence_states_secondTree[2, :]
                        ax.plot(x_vals_secondTree, y_vals_secondTree, z_vals_secondTree, 'r+', linewidth=0.5)

                if realTimeDraw:
                    plt.draw()
                    plt.pause(0.01)

                # Save the new valid primitives
                for neighbor_secondTree, cost_path_secondTree in last_states_secondTree:
                    
                    # Calculate tentative g score
                    tentative_g_cost_secondTree = current_g_secondTree + cost_path_secondTree
                    
                    # Update the hierarchy
                    if Node(neighbor_secondTree) not in g_cost_secondTree or tentative_g_cost_secondTree < g_cost_secondTree[Node(neighbor_secondTree)]:

                        # Save g cost in the g_cost dictionary
                        g_cost_secondTree[Node(neighbor_secondTree)] = tentative_g_cost_secondTree              
                        
                        # Compute the f_cost
                        f_cost_secondTree = calculate_f(neighbor_secondTree, map_instance, tentative_g_cost_secondTree,  heuristic(neighbor_secondTree, (map_instance["start_pos"][0], map_instance["start_pos"][1], map_instance["start_pos"][2])), dec, typeF_function, 2) 

                        # Add node dependency on current node
                        parents_dictionary_secondTree[Node(neighbor_secondTree)] = (current_node_secondTree)   

                        # Save the last node of the primitive along its f_cost
                        heapq.heappush(open_set_secondTree, (f_cost_secondTree, Node(neighbor_secondTree)))   

    # If we arrived here, no solution was found
    print("No solution found!")

    return [], 0, "maxIterations" # No path found 

def a_star_search(ax, plt, map_instance, realTimeDraw, typeF_function, dec):
    """
    This is the main function of the algorithm. This function runs the main loop for generating the path.
    If needed, change the initial condition of SAM in the "SAM initial state".
    If you want to change the starting position (or goal position), use MapGeneration_MotionPrimitives.py.

    This function returns (trajectory, successfulZeroOrOne, totalCost)
    """

    # Initialise general variables (valid for both trees)
    random.seed()
    sim = SAM_PRIMITIVES()
    dt_resolution = glbv.RESOLUTION_DT
    flag = 0    # for number of iterations
    nMaxIterations = 300

    # First tree variables
    x0 = map_instance["initial_state"]
    start = Node(x0)
    open_set = []                   # (cost_node, Node)
    parents_dictionary = {}         # (Node_current: Node_parent)
    g_cost = {start: 0}             # keeps track of the costs of the nodes, useful to understand more convenient paths
    resolution_dictionary = {}
    heapq.heappush(open_set, (0, start))
    parents_dictionary[start] = None
    arrivedPoint = False

    # Start the search
    while (open_set): 
        
        # Reconstruct the path of first tree if arrived to the goal
        if arrivedPoint:
            print("A star (first tree) ended successfully!")
            glbv.ARRIVED_PRIM = 0
            path, successfulSearch = reconstruct_path(Node(finalLast), parents_dictionary, resolution_dictionary, map_instance, ax, plt)

            if successfulSearch:
                return path, successfulSearch, "success"
        
        # Print the iteration number
        flag = flag + 1
        print(f"iteration {flag:.0f}")

        # Get the current node for the first tree (the one with cheapest f_cost in open_set)
        _, current_node = heapq.heappop(open_set)   #removes and returns the node with lowest f value
        current_g = g_cost[current_node]

        # Stop the algorithm if we exceed the maximum number of iterations
        if flag > nMaxIterations:
            break

        # Find new neighbors (last point of the primitives) using the motion primitives
        reached_states, last_states, arrivedPoint, final = get_neighbors(current_node, sim, map_instance, 1)
        finalLast = final[0] # in case we arrived
        finalCost = final[1] # in case we arrived 

        #If all the generated primitives are not in the free space, then continue with the next vertex
        if len(reached_states) == 0: 
            continue
        
        # Analyze the single steps within the primitives (each dt) for first tree
        for sequence_states in reached_states:

            # Save specific points for resolution of the trajectory
            list_vertices = getResolution(sequence_states, dt_resolution)
            resolution_dictionary[Node(sequence_states[:,-1])] = list_vertices

            # Plot the found motion primitives
            if realTimeDraw:
                x_vals = sequence_states[0, :]
                y_vals = sequence_states[1, :]
                z_vals = sequence_states[2, :]
                ax.plot(x_vals, y_vals, z_vals, 'c+', linewidth=0.5)

        if realTimeDraw:
            plt.draw()
            plt.pause(0.01)

        # Save the new valid primitives
        for neighbor, cost_path in last_states:
            
            # Calculate tentative g score
            tentative_g_cost = current_g + cost_path
            
            # Update the hierarchy
            if Node(neighbor) not in g_cost or tentative_g_cost < g_cost[Node(neighbor)]:

                # Save g cost in the g_cost dictionary
                g_cost[Node(neighbor)] = tentative_g_cost              
                
                # Compute the f_cost
                f_cost = calculate_f(neighbor, map_instance, tentative_g_cost,  heuristic(neighbor, (map_instance["goal_pixel"][0], map_instance["goal_pixel"][1], map_instance["goal_pixel"][2])), dec, typeF_function, 1)

                # Add node dependency on current node
                parents_dictionary[Node(neighbor)] = (current_node)   

                # Save the last node of the primitive along its f_cost
                heapq.heappush(open_set, (f_cost, Node(neighbor)))   
        
    # If we arrived here, no solution was found
    print("No solution found!")

    return [], 0, "maxIterations" # No path found 


# Possible results for "failingNotes":
# "success", "allOptimizationsFailed", "maxIterations", "NoPointsToConnect", "maxNumberOptimizations"