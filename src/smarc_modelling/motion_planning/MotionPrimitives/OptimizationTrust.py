from scipy.optimize import minimize, Bounds, NonlinearConstraint
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from smarc_modelling.vehicles.SAM import SAM
from smarc_modelling.motion_planning.MotionPrimitives.ObstacleChecker import *
import smarc_modelling.motion_planning.MotionPrimitives.GlobalVariables as glbv
import matplotlib.pyplot as plt
import os
import platform

# Provided AUV dynamics function (Fossen's model)
def fossen_dynamics(state, control_input):
    """Returns the next state given the current state and control input."""
    dt = glbv.RESOLUTION_DT
    sam = SAM(dt)
    #print(">>> >>> >>> current_state:")
    #print(state)
    #print(">>> >>> >>> control input:")
    #print(control_input)

    next_state_dot = sam.dynamics(state, control_input)

    # Forward Euler
    next_state = state + next_state_dot * dt

    # Midpoint Method (RK2)
    k2 = sam.dynamics(state+0.5*dt*next_state_dot, control_input)
    next_state = state + k2*dt

    '''
    if np.abs(next_state[0]) > 100 or np.abs(next_state[1]) > 100 or np.abs(next_state[2]) > 100:
        print(">>> >>> >>> current_state:")
        print(state)
        print(">>> >>> >>> control input:")
        print(control_input)
    '''

    # Check if the next state is infeasible
    #if IsOutsideTheMap(next_state[0], next_state[1], next_state[2], glbv.MAP_INSTANCE):
        #print("RuntimeWarning: a state returned by the fossen_dynamics function is odd")
        #return state, True  # Signal infeasibility
    
    return next_state, False

def clear_terminal():
    """Clears the terminal screen based on the operating system."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def objective_function(U_flat, x_init, N, map_instance):

    U = U_flat.reshape(N, 6)
    state = x_init.copy()
    sum_velocities = 0
    for i in range(N):
        state, errorHandler = fossen_dynamics(state, U[i])
        #if errorHandler:
        #    return 1e6  # Large penalty for infeasibility
        sum_velocities += np.linalg.norm(state[7:10])

    #final_velocity_magnitude = np.linalg.norm(state[7:10])

    return  sum_velocities #final_velocity_magnitude

def rpm_equality_constraint(U_flat, N):
    """
    Constraint to ensure rpm1 == rpm2 for all timesteps
    """

    U = U_flat.reshape(N, 6)

    # Return an array where each element is the difference between rpm1 and rpm2 for each timestep
    return U[:, 4] - U[:, 5]
    
def final_position_either_or_constraint(U_flat, x_init, N, map_instance):
    """
    Constraint to ensure that at least one of pointA or pointB (computed from the final state)
    is inside the goal region.
    
    U_flat: Flattened control input array.
    x_init: Initial state.
    N: Number of steps.
    goal_bounds: Dictionary containing goal boundary limits.
    """
    U = U_flat.reshape(N, 6)
    state = x_init.copy()

    # Simulate the trajectory
    for i in range(N):
        state, errorHandler = fossen_dynamics(state, U[i])

    # Compute final points A and B
    pointA = compute_A_point_forward(state)
    pointB = compute_B_point_backward(state)

    # Extract goal boundaries
    TILESIZE = map_instance["TileSize"]
    x_min = TILESIZE * map_instance["goal_area"][1]
    x_max = x_min + TILESIZE
    y_min = TILESIZE * map_instance["goal_area"][0]
    y_max = y_min + TILESIZE
    z_min = TILESIZE * map_instance["goal_area"][2]
    z_max = z_min + TILESIZE


    # Constraint values for each point
    g_A = min(x_max - pointA[0], pointA[0] - x_min, 
              y_max - pointA[1], pointA[1] - y_min, 
              z_max - pointA[2], pointA[2] - z_min)

    g_B = min(x_max - pointB[0], pointB[0] - x_min, 
              y_max - pointB[1], pointB[1] - y_min, 
              z_max - pointB[2], pointB[2] - z_min)

    # Final constraint value (at least one should be inside)
    return min(max(g_A, 0), max(g_B, 0))

def final_velocity_constraint(U_flat, x_init, N, map_instance):
    """Calculates 0.1 - norm(final velocity). Should be >= 0."""
    U = U_flat.reshape(N, 6)
    states = [x_init.copy()]
    state = x_init.copy()

    for u in U:
        state, errorHandler = fossen_dynamics(state, u)
        states.append(state)
    
    return 0.1 - np.linalg.norm(states[-1][7:10])

def position_constraint(U_flat, x_init, N, dim, lower_bound, upper_bound):
    """Constraint function to ensure the AUV stays within bounds in a given dimension (x, y, or z)."""
    U = U_flat.reshape(N, 6)  # Reshape flat control vector into (N, 6)
    state = x_init.copy()
    violations = []

    for i in range(N):
        state, errorHandler = fossen_dynamics(state, U[i])  # Propagate state
        pos = state[dim]  # Extract position along the given dimension

        # Two constraints: ensuring lower and upper bounds
        violations.append(pos - lower_bound)  # Must be >= 0
        violations.append(upper_bound - pos)  # Must be >= 0

    return np.array(violations)  # Returning an array allows checking all steps

def optimize_waypoints(x_init, U_init, N, map_instance):
    """Optimize control inputs to ensure zero final velocity using trust-constr."""

    control_dim = 6  # Number of control inputs per step
    init_guess = U_init.flatten()  # Flatten (N, 6) -> (N * 6,)

    # Define bounds for each control input
    control_bounds = [
        (10, 90),        # Vbs
        (10, 90),        # lcg
        (-np.deg2rad(6), np.deg2rad(6)),  # ds
        (-np.deg2rad(6), np.deg2rad(6)),  # dr
        (-1300, 1300),   # rpm1
        (-1300, 1300)    # rpm2
    ]

    # Convert bounds to Bounds object
    lower_bounds, upper_bounds = zip(*control_bounds)
    bounds = Bounds(np.tile(lower_bounds, N), np.tile(upper_bounds, N))

    # Position constraints
    x_min, y_min, z_min = 0, 0, 0
    x_max, y_max, z_max = map_instance["x_max"], map_instance["y_max"], map_instance["z_max"]

    # Convert constraints to NonlinearConstraint format
    final_velocity_constraint_nl = NonlinearConstraint(
        lambda U: final_velocity_constraint(U, x_init, N, map_instance),
        lb=0, ub=np.inf  # Should be >= 0
    )

    position_constraints_x = NonlinearConstraint(
        lambda U: position_constraint(U, x_init, N, 0, x_min, x_max),
        lb=0, ub=np.inf
    )

    position_constraints_y = NonlinearConstraint(
        lambda U: position_constraint(U, x_init, N, 1, y_min, y_max),
        lb=0, ub=np.inf
    )

    position_constraints_z = NonlinearConstraint(
        lambda U: position_constraint(U, x_init, N, 2, z_min, z_max),
        lb=0, ub=np.inf
    )

    final_position_constraint_nl = NonlinearConstraint(
        lambda U: final_position_either_or_constraint(U, x_init, N, map_instance),
        lb=0, ub=np.inf  # At least one point should be inside the goal area
    )

    rpm_equality_constraint_nl = NonlinearConstraint(
        lambda U: rpm_equality_constraint(U, N),
        lb=0, ub=0  # Equality constraint
    )

    # Optimize using trust-constr
    result = minimize(
        objective_function, init_guess, args=(x_init, N, map_instance),
        method="trust-constr", bounds=bounds,
        constraints=[final_velocity_constraint_nl, position_constraints_x, position_constraints_y,
                     position_constraints_z, final_position_constraint_nl, rpm_equality_constraint_nl],
        options={"maxiter": 50, 'xtol': 1e-9, 'verbose': 2}
    )

    best_U = result.x.reshape(N, control_dim)

    # Simulate the trajectory
    states = [x_init.copy()]
    state = x_init.copy()
    for u in best_U:
        state = fossen_dynamics(state, u)
        states.append(state)

    global_v = body_to_global_velocity((states[-1][3], states[-1][4], states[-1][5], states[-1][6]), [states[-1][7], states[-1][8], states[-1][9]])
    print(f"last velocity norm:...{np.linalg.norm(global_v[0:3])} m/s")
    print(f"IsLastStateInGoal?...{arrived(states[-1], map_instance)}")
    return states, best_U

def testOptimization(waypoints, map_instance, axx, pltt):

    glbv.MAP_INSTANCE = map_instance
    
    N = len(waypoints) - 1  # Number of control intervals
    if N <= 0:
        print("Error: Need at least two waypoints for optimization.")
        return

    # Extract control inputs from waypoints (assuming waypoints contain state and control)
    x0 = waypoints[0]
    U_init = np.zeros((N, 6))
    for i in range(N):
        # Assuming the last 6 elements of each waypoint are the control inputs
        U_init[i] = waypoints[i][13:19]

    optimal_states, optimal_controls = optimize_waypoints(x0, U_init, N, map_instance)

    x_coords = [state[0] for state in optimal_states]
    y_coords = [state[1] for state in optimal_states]
    z_coords = [state[2] for state in optimal_states]

    # Plot the trajectory in 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x_coords, y_coords, z_coords)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Optimized AUV Trajectory')
    plt.draw()
    #plt.show()

    #final_list = []
    #for i in range(N // 2):
    #    final_list.append(waypoints[i])

    #return final_list + optimal_states
    return optimal_states

