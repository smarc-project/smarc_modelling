import casadi as ca
import numpy as np
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi
from smarc_modelling.MotionPrimitives.ObstacleChecker_MotionPrimitives import *
import smarc_modelling.MotionPrimitives.GlobalVariables_MotionPrimitives as glbv

def fossen_dynamics_casadi(state, control_input):
    """
    Returns the next state given the current state and control input (CasADi version)
    """

    # Initialize variables
    dt = glbv.RESOLUTION_DT
    sam = SAM_casadi(dt)
    
    # Compute state derivative using RK2
    symbolic_function = sam.dynamics()
    next_state_dot = symbolic_function(state, control_input)
    next_state = state + dt * next_state_dot

    variable = sam.export_dynamics_model()
    #k2 = symbolic_function(state + 0.5 * dt * next_state_dot, control_input)
    #next_state = state + k2 * dt
    #print(next_state[0])
    # Return next_state
    return next_state

def simple_dynamics_casadi(state, control_input):
    """A very simple linear dynamics model for testing."""
    dt = glbv.RESOLUTION_DT
    # Assuming a state vector where the first few elements are position
    # and the next few are velocity.
    # Simple model: position changes with velocity, velocity changes with control.
    next_state = ca.MX(state.shape[0], 1)
    next_state[0:3] = state[0:3] + dt * state[7:10]  # Position update (x, y, z)
    next_state[7:10] = state[7:10] + dt * control_input[0:3]  # Velocity update (vx, vy, vz)
    # The rest of the state remains unchanged for simplicity
    next_state[3:7] = state[3:7]  # Quaternion
    next_state[10:] = state[10:] # Other states

    return next_state

def define_optimization_casadi(waypoints, map_instance, N, nx, nu):
    """
    Define and solve the optimization problem in CasADi
    """
    
    # Initialise the optimization problem
    opti = ca.Opti() 
    
    # Initialise the state variables (X) and control inputs (U)
    X = opti.variable(nx, N + 1)  
    U = opti.variable(nu, N)  
    #x = opti.variable()
    #y = opti.variable()
    
    # Initialise the x0 parameter
    x0 = opti.parameter(nx) 
    opti.set_value(x0, waypoints[0])

    # Define the objective function: minimize final velocity norm
    
    obj = X[0, N-1]
    opti.minimize(obj)

    #obj = 0
    #for k in range(N):
    #    obj += ca.mtimes([U[:, k].T, Q, U[:, k]])  # Quadratic cost on control
    #obj += ca.mtimes([X[-1, :].T, P, X[-1, :]])  # Cost on final velocity

    #for k in range(N):
    #    obj += ca.norm_2(X[7:10, k+1])  # Sum velocity magnitudes

    # Constraint: initial state constraint
    opti.subject_to(X[:, 0] == x0)
    #opti.subject_to(ca.norm_2(X[3:7]) == 1)
    # Constraint: dynamics constraint and RPM constraint
    
    for k in range(N):

        # Dynamics
        x_next = fossen_dynamics_casadi(X[:, k], U[:, k])
        opti.subject_to(X[:, k+1] == x_next)
        
        """
        # RPM 
        opti.subject_to(U[4, k] == U[5, k])
        """
    
    '''
    # Constraint: final position (at least one between A and B) inside the goal area
    state_N = X[:, -1]  # Extract final position
    pointA = compute_A_point_forward_casadi(state_N, opti)
    pointB = compute_B_point_backward_casadi(state_N, opti)
    
    TILESIZE = map_instance["TileSize"]
    x_min, x_max = TILESIZE * map_instance["goal_area"][1], TILESIZE * (map_instance["goal_area"][1] + 1)
    y_min, y_max = TILESIZE * map_instance["goal_area"][0], TILESIZE * (map_instance["goal_area"][0] + 1)
    z_min, z_max = TILESIZE * map_instance["goal_area"][2], TILESIZE * (map_instance["goal_area"][2] + 1)
    
    g_A = ca.mmin(ca.vertcat(x_max - pointA[0], pointA[0] - x_min,
                         y_max - pointA[1], pointA[1] - y_min,
                         z_max - pointA[2], pointA[2] - z_min))

    g_B = ca.mmin(ca.vertcat(x_max - pointB[0], pointB[0] - x_min,
                         y_max - pointB[1], pointB[1] - y_min,
                         z_max - pointB[2], pointB[2] - z_min))
    opti.subject_to(ca.mmax(ca.vertcat(g_A, g_B)) >= 0)  # At least one must be inside
    '''

    # Constraints: control bounds for the input
    opti.subject_to(opti.bounded(10, U[0, :], 90))  # Vbs
    opti.subject_to(opti.bounded(10, U[1, :], 90))  # lcg
    opti.subject_to(opti.bounded(-np.deg2rad(6), U[2, :], np.deg2rad(6)))  # ds
    opti.subject_to(opti.bounded(-np.deg2rad(6), U[3, :], np.deg2rad(6)))  # dr
    opti.subject_to(opti.bounded(-1300, U[4, :], 1300))  # rpm1
    opti.subject_to(opti.bounded(-1300, U[5, :], 1300))  # rpm2

    # Solver selection
    opti.solver('ipopt')

    # Return 
    return opti, X, U

def solve_optimization(waypoints, map_instance):
    """
    Solve the CasADi optimization problem
    """

    # Define dimensions
    N = len(waypoints) - 1  # Number of control intervals
    nx = len(waypoints[0])  # State dimension
    nu = 6  # Control input dimension

    # Get opti
    opti, X, U = define_optimization_casadi(waypoints, map_instance, N, nx, nu)

    # Set initial control guess from waypoints
    U_init = np.zeros((nu, N))
    for i in np.arange(0, N):
        U_init[:, i] = waypoints[i][13:19]
    opti.set_initial(U, U_init)

    try:
        sol = opti.solve()
        #print(f"first waypoint:...{waypoints[0]}")
        #print(f"solution:...{sol.value(X[:,0])}")
        print("Solution found!")
    except RuntimeError as e:
        print("Solver failed!")
        #print(opti.debug.value(X))
        raise e


    # Solve
    sol = opti.solve()
    
    # Copy optimal states
    optimal_states = sol.value(X)
    optimal_controls = sol.value(U)

    # Return states
    #return optimal_states, optimal_controls
    return waypoints

def compute_A_point_forward_casadi(state, opti, distance=0.655):
    """
    Compute the point 0.655 meters forward along the vehicle's longitudinal axis (symbolic)
    """

    # Get the variables
    x, y, z = state[0], state[1], state[2]
    q0, q1, q2, q3 = state[3], state[4], state[5], state[6]

    # Rotation matrix from quaternion (CasADi symbolic way)
    R_b2w = ca.vertcat(
        ca.horzcat(1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)),
        ca.horzcat(2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)),
        ca.horzcat(2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2))
    )

    # Get the forward direction of the robot 
    forward_body = ca.MX([1, 0, 0])  # Forward direction in body frame
    forward_world = R_b2w @ forward_body  # Transform to world frame
    forward_world = forward_world / ca.norm_2(forward_world)

    # Get the new point
    new_point = ca.vertcat(x, y, z) + distance * forward_world

    # Return it
    return new_point

def compute_B_point_backward_casadi(state, opti, distance=0.655):
    """
    Compute the point 0.655 meters backward along the vehicle's longitudinal axis (symbolic)
    """

    # Get the variables
    x, y, z = state[0], state[1], state[2]
    q0, q1, q2, q3 = state[3], state[4], state[5], state[6]

    # Rotation matrix from quaternion (CasADi symbolic way)
    R_b2w = ca.vertcat(
        ca.horzcat(1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)),
        ca.horzcat(2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)),
        ca.horzcat(2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2))
    )

    # Get the forward direction of the robot
    forward_body = ca.MX([1, 0, 0])  # Forward direction in body frame
    forward_world = R_b2w @ forward_body  # Transform to world frame
    forward_world = forward_world / ca.norm_2(forward_world)

    # Get the new point
    new_point = ca.vertcat(x, y, z) - distance * forward_world

    # Return it
    return new_point
