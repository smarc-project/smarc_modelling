import numpy as np
from casadi import SX, vertcat, sqrt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from smarc_modelling.control.control import *
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
def create_ocp(model, x0, x_last, N, map_instance):

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
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
    ocp.solver_options.qp_tol = 1e-6       # QP tolerance
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.regularize_method = 'NO_REGULARIZE'
    
    # CURRENT COST: Cost function
    # Cost function: minimize final velocity + sum(k=0, N){u_k^T Q u_k}
    #nu = model.u.rows()
    nx = model.x.rows()
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    #ocp.cost.cost_type = 'NONLINEAR_LS'
    # Define the Q matrix
    Q_diag = np.ones(19)
    Q_diag[ 0:3 ] = 12e1         # Position
    Q_diag[ 3:7 ] = 12e1         # Quaternion
    Q_diag[ 7:10] = 1e1        # linear velocity
    Q_diag[10:13] = 1e1         # Angular velocity
    Q_diag[13:15] = 1e-4        # VBS, LCG
    Q_diag[15:17] = 1/200     # stern_angle, rudder_angle
    Q_diag[17:  ] = 1e-6        # RPM1 And RPM2
    Q_diag[13:  ] = Q_diag[13:  ]
    Q = np.diag(Q_diag)
    ocp.cost.W_e = Q
    #Q = np.diag([1e-4, 1e-4, 1/200, 1/200, 1e-6, 1e-6]) # Costs set according to Bryson's rule (MPC course) 
    #Q = np.diag([1e-6, 1e-6]) 
    #ocp.cost.W = Q
    ocp.cost.yref_e = x_last
    #ocp.cost.yref = np.array([0, 0])
    #ocp.cost.yref = np.array([0, 0, 0])
    ocp.model.cost_y_expr_e = model.x
    #ocp.model.cost_y_expr = model.x[17:19] 

    # Constraints
    ocp.constraints.x0 = x0
    
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

    '''
    constraint_RPM2 = vertcat(
        model.x[17] - model.x[18]
    )
    
    # For each step
    ocp.model.con_h_expr = vertcat(constraint_RPM2)
    ocp.constraints.lh = np.array([
        0
    ])
    ocp.constraints.uh = np.array([
        0
    ])
    '''
    
    # At the end
    '''
    ocp.model.con_h_expr_e = vertcat(goal_constraints_pointA)
    ocp.constraints.lh_e = np.array([
        arrivalx_min, arrivaly_min, arrivalz_min  # cg bounds
    ])
    ocp.constraints.uh_e = np.array([
        arrivalx_max, arrivaly_max, arrivalz_max  # cg bounds
    ])
    '''

    '''
    # Final constraint in goal and z
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

    # Set constraints on the final state
    ocp.constraints.lbx_e = x_lbx
    ocp.constraints.ubx_e = x_ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    # Return ocp
    return ocp

def optimization_acados_doubleTree(waypoints, map_instance):
    # Define class and model
    dt = glbv.RESOLUTION_DT
    N = len(waypoints)
    sam = SAM_casadi(dt)
    nmpc = NMPC_trajectory(sam, dt, N)
    model = nmpc.export_dynamics_model(sam)

    # Create ocp
    ocp = create_ocp(model, waypoints[0], waypoints[-1], len(waypoints), map_instance)

    # Solver setup
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    # Set initial guess from waypoints
    '''
    for i, waypoint in enumerate(waypoints):
        if i > ocp.dims.N:
            break
        ocp_solver.set(i, "x", np.array(waypoint))
    '''

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

    # Return optimized waypoints
    return optimized_waypoints, status
