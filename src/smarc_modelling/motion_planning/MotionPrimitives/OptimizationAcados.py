import numpy as np
from casadi import SX, vertcat, sqrt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from smarc_modelling.vehicles.SAM_casadi import *
import smarc_modelling.motion_planning.MotionPrimitives.GlobalVariables as glbv

from casadi import SX, MX, vertcat, sqrt, horzcat

def compute_A_point_forward_casadi(state, distance=0.655):
    """
    Compute the point forward along the vehicle's longitudinal axis using CasADi.
    """
    # Get current state elements
    x = state[0]
    y = state[1]
    z = state[2]
    q0 = state[3]
    q1 = state[4]
    q2 = state[5]
    q3 = state[6]

    # Normalize quaternion
    norm_q = sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0 /= norm_q
    q1 /= norm_q
    q2 /= norm_q
    q3 /= norm_q

    # Forward direction in body frame (longitudinal axis)
    forward_body = vertcat(1, 0, 0)  # X-axis in body frame

    # Rotation matrix from quaternion
    R = vertcat(
        horzcat(1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)),
        horzcat(2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)),
        horzcat(2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2))
    )

    # Transform to world frame
    forward_world = R @ forward_body

    # Normalize forward vector
    forward_norm = sqrt(forward_world[0]**2 + forward_world[1]**2 + forward_world[2]**2)
    forward_world /= forward_norm

    # Compute new point
    new_point = vertcat(x, y, z) + distance * forward_world

    return new_point

def compute_B_point_backward_casadi(state, distance=0.655):
    """
    Compute the point backward along the vehicle's longitudinal axis using CasADi.
    """
    # Get current state elements
    x = state[0]
    y = state[1]
    z = state[2]
    q0 = state[3]
    q1 = state[4]
    q2 = state[5]
    q3 = state[6]

    # Normalize quaternion
    norm_q = sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0 /= norm_q
    q1 /= norm_q
    q2 /= norm_q
    q3 /= norm_q

    # Forward direction in body frame (longitudinal axis)
    forward_body = vertcat(1, 0, 0)  # X-axis in body frame

    # Rotation matrix from quaternion
    R = vertcat(
        horzcat(1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)),
        horzcat(2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)),
        horzcat(2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2))
    )

    # Transform to world frame
    forward_world = R @ forward_body

    # Normalize forward vector
    forward_norm = sqrt(forward_world[0]**2 + forward_world[1]**2 + forward_world[2]**2)
    forward_world /= forward_norm

    # Compute new point (backward)
    new_point = vertcat(x, y, z) - distance * forward_world

    return new_point

# Create an OCP object
def create_ocp(model, x0, N, map_instance):

    # Initialization Acados + options
    ocp = AcadosOcp()
    ocp.model = model
    ts = glbv.RESOLUTION_DT
    N_horizon = N
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon*ts  
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode = 'ROBUST'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 3 #3 default
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 80
    ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
    ocp.solver_options.qp_tol = 1e-6       # QP tolerance
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'
    
    # CURRENT COST: final v + sum U (not angular velocity)
    # Cost function: minimize final velocity + sum(k=0, N){u_k^T Q u_k}
    #nu = model.u.rows()
    nx = model.x.rows()
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.W_e = np.diag([1, 1, 1])
    #Q = np.diag([1e-4, 1e-4, 1/200, 1/200, 1e-6, 1e-6]) # Costs set according to Bryson's rule (MPC course) 
    Q = np.diag([5e-7, 5e-7]) # Costs set according to Bryson's rule (MPC course) 
    ocp.cost.W = Q
    ocp.cost.yref_e = np.array([0, 0, 0])
    #ocp.cost.yref = np.array([0, 0, 0, 0, 0, 0])
    ocp.cost.yref = np.array([0, 0])
    ocp.model.cost_y_expr_e = model.x[7:10]
    ocp.model.cost_y_expr = model.x[17:19] 

    # Constraints
    ocp.constraints.x0 = x0
    
    # Terminal state constraints: Goal area bounds
    TILESIZE = map_instance["TileSize"]
    arrivalx_min = TILESIZE * map_instance["goal_area"][1]
    arrivalx_max = arrivalx_min + TILESIZE
    arrivaly_min = TILESIZE * map_instance["goal_area"][0]
    arrivaly_max = arrivaly_min + TILESIZE
    arrivalz_min = TILESIZE * map_instance["goal_area"][2]
    arrivalz_max = arrivalz_min + TILESIZE
    match map_instance["where"]:
        case "top":
            x_g_front_min = TILESIZE * map_instance["goal_area"][1]
            x_g_front_MAX = x_g_front_min + TILESIZE
            y_g_front_min = TILESIZE * (map_instance["goal_area"][0] - 2)
            y_g_front_MAX = y_g_front_min + 5*TILESIZE
            z_g_front_min = TILESIZE * (map_instance["goal_area_front"][2]) - 0.1
            z_g_front_MAX = z_g_front_min + TILESIZE + 0.2
        case _:
            x_g_front_min = TILESIZE * (map_instance["goal_area"][1] - 2)
            x_g_front_MAX = x_g_front_min + 5*TILESIZE
            y_g_front_min = TILESIZE * map_instance["goal_area"][0]
            y_g_front_MAX = y_g_front_min + TILESIZE
            z_g_front_min = TILESIZE * (map_instance["goal_area_front"][2]) - 0.1
            z_g_front_MAX = z_g_front_min + TILESIZE + 0.2

    # Define terminal state constraints as symbolic expressions
    
    goal_constraints_cg = vertcat(
        model.x[0],  # x position
        model.x[1],  # y position
        model.x[2]   # z position
    )

    # Goal constraint for front of SAM
    pointA = compute_A_point_forward_casadi(model.x)
    pointB = compute_B_point_backward_casadi(model.x)
    goal_constraints_pointA = vertcat(
        pointA[0],
        pointA[1],
        pointA[2]
    )
    constraints_point_B = vertcat(
        pointB[0],
        pointB[1],
        pointB[2]
    )
    goal_constraint_z = vertcat(
        pointA[2] - pointB[2]
    )
    # Combine both constraints
    ocp.model.con_h_expr_e = vertcat(goal_constraints_cg, goal_constraints_pointA)

    # Combine bounds for both constraints
    ocp.constraints.lh_e = np.array([
        x_g_front_min, y_g_front_min, z_g_front_min,  # cg bounds
        x_g_front_min, y_g_front_min, z_g_front_min  # pointA bounds
    ])
    ocp.constraints.uh_e = np.array([
        x_g_front_MAX, y_g_front_MAX, z_g_front_MAX,  # cg bounds
        x_g_front_MAX, y_g_front_MAX, z_g_front_MAX  # pointA bounds
    ])

    # Final constraint in goal and z
    '''
    ocp.model.con_h_expr_e = vertcat(goal_constraints_cg, goal_constraint_z)
    ocp.constraints.lh_e = np.array([
        arrivalx_min, arrivaly_min, arrivalz_min,
        -0.1
    ])
    ocp.constraints.uh_e = np.array([
        arrivalz_max, arrivaly_max, arrivalz_max,
        0.1
    ])
    '''

    # Constraint: x in XFREE
    xMax = map_instance["x_max"] 
    yMax = map_instance["y_max"] 
    zMax = map_instance["z_max"] 

    ocp.model.con_h_expr = vertcat(goal_constraints_pointA, constraints_point_B)
    ocp.constraints.lh = np.array([
        0, 0, 0, 
        0, 0, 0
    ])
    ocp.constraints.uh = np.array([
        xMax, yMax, zMax, 
        xMax, yMax, zMax
    ])

    # Set constraints on the states
    x_ubx = np.ones(nx)
    x_ubx[  :13] = 1000

    # Set bounds on the state and inputs
    x_ubx[13:15] = 100 
    x_ubx[15:17] = np.deg2rad(7)
    x_ubx[17:  ] = 1300
    x_lbx = -x_ubx
    x_lbx[13:15] = 0
    ocp.constraints.lbx = x_lbx
    ocp.constraints.ubx = x_ubx
    ocp.constraints.idxbx = np.arange(nx)

    # Return ocp
    return ocp

def optimization_acados(waypoints, map_instance):
    # Define class and model
    sam = SAM_casadi()
    model = sam.export_dynamics_model()

    # Create ocp
    ocp = create_ocp(model, waypoints[0], len(waypoints), map_instance)

    # Solver setup
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    # Set initial guess from waypoints
    for i, waypoint in enumerate(waypoints):
        if i > ocp.dims.N:
            break
        ocp_solver.set(i, "x", np.array(waypoint))
    

    # Solve the problem
    status = ocp_solver.solve()
    if status != 0:
        print(f"Solver failed with status {status}")
    else:
        print("Optimization successful!")

    # Extract the optimized waypoints and save them
    optimized_waypoints = []
    for i in range(ocp.dims.N + 1):
        x_opt = ocp_solver.get(i, "x")
        optimized_waypoints.append(x_opt)
        #print(f"Step {i}, x: {x_opt}")

    # Return optimized waypoints
    return optimized_waypoints
