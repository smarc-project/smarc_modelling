import heapq
import numpy as np
import sys
import random
from smarc_modelling.MotionPrimitives.MotionPrimitives_MotionPrimitives import SAM_PRIMITIVES
from smarc_modelling.MotionPrimitives.ObstacleChecker_MotionPrimitives import calculate_angle_betweenVectors, calculate_angle_goalVector
from smarc_modelling.MotionPrimitives.OptimizationTrust_MotionPrimitives import testOptimization
from smarc_modelling.MotionPrimitives.acados_Trajectory_simulator import MPC_optimization
from smarc_modelling.MotionPrimitives.OptimizationAcados_MotionPrimitives import optimization_acados
import smarc_modelling.MotionPrimitives.GlobalVariables_MotionPrimitives as glbv
from joblib import Parallel, delayed
from threading import Lock
from scipy.spatial.transform import Rotation as R
import time
import multiprocessing
import pandas as pd
import csv

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

def process_result(result, shared_data, lock):
    data, last_state, inObstacle, arrived = result
    
    #if arrived:
    if not inObstacle:
        last = last_state[0]
        q0, q1, q2, q3, v1, v2, v3 = last[3:10]
        final_v_norm = np.linalg.norm(body_to_global_velocity((q0, q1, q2, q3), [v1, v2, v3]))

        # First check outside lock to avoid unnecessary contention
        if final_v_norm < shared_data.best_v_norm and not inObstacle:
            with lock:  # Acquire lock only if condition is met
                if final_v_norm < shared_data.best_v_norm:  # Double-check inside lock
                    shared_data.best_v_norm = final_v_norm
                    shared_data.bestData = data

def parallel_process(results):
    """ 
    Set up multiprocessing pool to process results in parallel
    """

    manager = multiprocessing.Manager()
    shared_data = manager.Namespace()  # Shared memory for best values
    shared_data.best_v_norm = float('inf')  # Initialize with high value
    shared_data.bestData = None  # Placeholder for best data

    lock = manager.Lock()  # Lock for synchronization

    # Use multiprocessing pool to parallelize
    with multiprocessing.Pool() as pool:
        pool.starmap(process_result, [(res, shared_data, lock) for res in results])

    return shared_data.bestData  # Return best trajectory data

def postProcessVelocity(list_vertices, map_instance, ax, plt):

    print("post processing for reducing final velocity")
    startT = time.time()
    sim = SAM_PRIMITIVES()
    len_list = len(list_vertices)
    first_vertex = list_vertices[int((len_list / 10) * 3)]
    #first_vertex = list_vertices[0]
    q0, q1, q2, q3, v1, v2, v3 = first_vertex[3:10]

    # 1 # Define the inputs 
    rudder_inputs = np.arange(-7, 7, 3)
    stern_inputs = np.array([-7, 0, 7])
    vbs_inputs = np.array([10, 50, 90])
    lcg_inputs = np.array([0, 50, 100])
    rpm_inputs = np.arange(-1500, 1600, 100)

    # 2 # Add the name of the input into np.meshgrid(), and change the second value of .reshape(., THIS)
    input_pairs = np.array(np.meshgrid(rudder_inputs, rpm_inputs, vbs_inputs, lcg_inputs, stern_inputs)).T.reshape(-1,5)

    # 3 # Add the index in u of the input you modified in np.tile([..., HERE], ...)
    additional_values = np.tile([3, 4, 0, 1, 2], (input_pairs.shape[0], 1))

    # 4 # Do not touch
    full_input_pairs = np.hstack((input_pairs, additional_values))

    # Generate primitives
    st = time.time()
    results = Parallel(n_jobs=multiprocessing.cpu_count())(
    delayed(process_input_pair)(inputs, first_vertex, sim, map_instance, True) for inputs in full_input_pairs
    ) 
    end = time.time()
    print(f"parallel time for generating primitives:...{end-st:.4f} seconds")

    # Run parallel processing for evaluating primitives
    st = time.time()
    bestData = parallel_process(results)
    end = time.time()
    print(f"Parallel time for evaluating primitives:...{end-st:.4f} seconds")

    # At least one successful trajectory
    list_vertices1 = [first_vertex]
    final_list_vertices = []
    if bestData is not None:
        list_vertices2 = getResolution(bestData, glbv.RESOLUTION_DT)
        final_list_vertices = list_vertices1 + list_vertices2

    stopT = time.time()
    print(f"Post-processing time:...{stopT-startT:.4f} seconds")

    return final_list_vertices

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
        if header:
            print(f"Header row: {header}")
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
            start_opti = time.time()
            print("<starting optimization>")
            # Load trajectory
            res_list = load_trajectory_from_csv('OptimizationTrajectory.csv')
            print(res_list)
            result_list = optimization_acados(res_list, map_instance)
            if len(result_list) > 0:
                res_list = result_list
            print("[     OK     ]")
            end_opti = time.time()
            print(f"optimization time:...{end_opti-start_opti:.4f} seconds")
            return res_list

def reconstruct_path(current, parents_dict, resolution_dict, map_instance, ax, plt):
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

        '''
        # Post processing velocity
        if len(final_path) == 0:
            result_list = postProcessVelocity(res_list, map_instance, ax, plt)
            if len(result_list) > 0:
                res_list = result_list
        '''

        '''
        # Optimization TRUST
        if len(final_path) == 0:
            start_opti = time.time()
            print("<starting optimization>")
            x0 = res_list[0]
            result_list = testOptimization(res_list, map_instance, ax, plt)
            if len(result_list) > 0:
                res_list = result_list
            print("[     OK     ]")
            end_opti = time.time()
            print(f"optimization time:...{end_opti-start_opti:.4f} seconds")
        '''

        
        # Optimization Acados
        if len(final_path) == 0:
            start_opti = time.time()
            print("<starting optimization>")
            #print(res_list)

            # Save trajectory
            #df = pd.DataFrame(res_list, columns=["x", "y", "z", "q0", "q1", "q2", "q3", "u", "v", "w", "q", "p", "r", "V_bs", "l_cg", "ds", "dr", "rpm_1", "rpm_2"])
            #df.to_csv("OptimizationTrajectory.csv", index=False)

            result_list = optimization_acados(res_list, map_instance)
            if len(result_list) > 0:
                res_list = result_list
            print("[     OK     ]")
            end_opti = time.time()
            print(f"optimization time:...{end_opti-start_opti:.4f} seconds")
              

        # Optimization MPC
        '''
        if len(final_path) == 0:
            start_opti = time.time()
            print("<starting optimization>")
            print(res_list)
            result_list = MPC_optimization(res_list, map_instance)
            if len(result_list) > 0:
                res_list = result_list
            print("[     OK     ]")
            end_opti = time.time()
            print(f"optimization time:...{end_opti-start_opti:.4f} seconds")
        '''

        # Append vertices to the final list (reverted order)
        for vertex in res_list[::-1]:
            final_path.append(vertex)

        # Update current 
        current = parents_dict[current]
    
    '''
    # Find the path from goal to start (OLD ONE)
    while current is not None:
        final_path.append(current.state)
        current = parents_dict[current]
    '''

    # Return reversed path
    return final_path[::-1] 

def process_input_pair(inputs, current_state, sim, map_instance, breaking=False):
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
    alpha = calculate_angle_goalVector(current_state, v_vector, map_instance)

    # Get all the points within one single input primitive
    data, cost, inObs, arrived, finalState = sim.curvePrimitives(current_state, inputs[0 : inputLen//2], inputs[inputLen//2 : inputLen], map_instance, alpha, breaking)
    if not inObs:
        neighbour = data[:,-1]
        cost_path = cost

        # Return (allPoints, (lastPoint, costPath), boolInObstacle, boolArrived, stateInGoal(or None))
        return data, (neighbour, cost_path), inObs, arrived  
    
    # If no valid primitive is available, return an empty array
    return np.array([]), ([], -1), inObs, arrived

def compute_current_orientationVector(state, map_inst):
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
    angle = calculate_angle_goalVector(state, forward_vector, map_inst)
    if angle < np.pi/2:
        orientation_vector = forward_vector
    else:
        orientation_vector = backward_vector

    return orientation_vector

def get_neighbors(current, sim, map_instance):
    """
    This function is used to compute the motion primitives for the current state.

    This function will return:
    1) A list containing all the valid primitives (all the states within all the valid primitives)
    2) A list containing only the last states of the valid primitives
    3) True/False based on if at least one primitive arrived at the goal
    4) The final state and cost if we arrived at the goal
    """

    '''
    # Dynamic step size
    q0, q1, q2, q3 = current.state[3:7]
    vx, vy, vz = body_to_global_velocity((q0, q1, q2, q3), current.state[7:10])
    v_vector = np.array([vx, vy, vz])
    v_vector_norm = np.linalg.norm(v_vector)
    if v_vector_norm != 0:
        v_vector /= v_vector_norm
    
    angle = calculate_angle_goalVector(current.state, v_vector, map_instance)
    if np.rad2deg(angle) < 10:
        dynamic_step = 3
    elif np.rad2deg(angle) < 40:
        dynamic_step = 2
    elif np.rad2deg(angle) < 75:
        dynamic_step = 1.5
    else:
        dynamic_step = 2
    '''
    
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

    start_p = time.time()

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
        delayed(process_input_pair)(inputs, current.state, sim, map_instance) for inputs in full_input_pairs
    ) 
    end_p = time.time()

    # Print computational time for the parallelization
    print(f"parallel time:...{end_p-start_p:.2f}")

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
                bestOrientation = compute_current_orientationVector(last_state[0], map_instance)
                finalAngle = calculate_angle_goalVector(last_state[0], bestOrientation, map_instance)
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

def calculate_f(neighbor, map_instance, tentative_g, heuristic_cost, dec, typeF):
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
                c = np.sqrt(goal_vector_norm**2 + d**2 - 2*goal_vector_norm*d*np.cos(angleBrake))
                heuristic = c + d
            else:
                heuristic = goal_vector_norm

            total_f = (tentative_g**2 + heuristic**2) / tentative_g
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

def a_star_search(ax, plt, map_instance, realTimeDraw, typeF_function, dec):
    """
    This is the main function of the algorithm. This function runs the main loop for generating the path.
    If needed, change the initial condition of SAM in the "SAM initial state".
    If you want to change the starting position (or goal position), use MapGeneration_MotionPrimitives.py.

    This function returns (trajectory, successfulZeroOrOne, totalCost)
    """

    # For randomisaztion in the function
    random.seed()

    # SAM initial state 
    eta0 = np.zeros(7)
    eta0[0] = map_instance["start_pos"][0]
    eta0[1] = map_instance["start_pos"][1]
    eta0[2] = map_instance["start_pos"][2]
    eta0[3] = 1
    eta0[4] = 0
    eta0[5] = 0
    eta0[6] = 0
    nu0 = np.zeros(6)   # Zero initial velocities
    u0 = np.zeros(6)    #The initial control inputs for SAM
    u0[0] = 50          #Vbs
    u0[1] = 50          #lcg
    x0 = np.concatenate([eta0, nu0, u0])
    
    # Initialize variables
    sim = SAM_PRIMITIVES()
    start = Node(x0)
    open_set = []                   # (cost_node, Node)
    parents_dictionary = {}         # (Node_current: Node_parent)
    g_cost = {start: 0}             # keeps track of the costs of the nodes, useful to understand more convenient paths
    resolution_dictionary = {}
    heapq.heappush(open_set, (0, start))
    parents_dictionary[start] = None
    flag = 0
    nMaxIterations = 1000
    arrivedPoint = False
    dt_resolution = glbv.RESOLUTION_DT

    # Start the search
    while (open_set):
        
        # Reconstruct the path if arrived to the goal
        if arrivedPoint:
            print("A star ended successfully!")
            glbv.ARRIVED_PRIM = 0
            #pre_processed_trajectory = reconstruct_path(Node(finalLast), parents_dictionary, resolution_dictionary, map_instance, ax, plt)
            #new_trajectory = MPC_optimization(pre_processed_trajectory, map_instance)
            return reconstruct_path(Node(finalLast), parents_dictionary, resolution_dictionary, map_instance, ax, plt), 1, finalCost
        
        # Print the iteration number
        flag = flag + 1
        print(f"Iteration {flag:.0f}.")

        # Get the current node (the one with cheapest f_cost in open_set)
        _, current_node = heapq.heappop(open_set)   #removes and returns the node with lowest f value
        current_g = g_cost[current_node]

        # Stop the algorithm if we exceed the maximum number of iterations
        if flag > nMaxIterations:
            break

        # Find new neighbors (last point of the primitives) using the motion primitives
        reached_states, last_states, arrivedPoint, final = get_neighbors(current_node, sim, map_instance)
        finalLast = final[0] # in case we arrived
        finalCost = final[1] # in case we arrived 

        # If all the generated primitives are not in the free space, then continue with the next vertex
        if len(reached_states) == 0: 
            continue
        
        # Analyze the single steps within the primitives (each dt)
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
                f_cost = calculate_f(neighbor, map_instance, tentative_g_cost,  heuristic(neighbor, (map_instance["goal_pixel"][0], map_instance["goal_pixel"][1], map_instance["goal_pixel"][2])), dec, typeF_function)

                # Add node dependency on current node
                parents_dictionary[Node(neighbor)] = (current_node)   

                # Save the last node of the primitive along its f_cost
                heapq.heappush(open_set, (f_cost, Node(neighbor)))   

    # If we arrived here, no solution was found
    print("No solution found!")

    return [], 0, 0 # No path found 