import sys
import numpy as np
import smarc_modelling.motion_planning.MotionPrimitives.GlobalVariables as glbv
from scipy.spatial.transform import Rotation as R

def IsWithinObstacle(x,y,z, map_instance):
    """
    This function checks if a PIXEL (x,y,z) is within an obstacle defined by obstacleDisctionary
    It returns True if (x,y,z) collides with an obstacle and False otherwise

    --> How to check if a pixel position (x,y) is within an obstacle: IsWithinObstacle(x,y)
    """

    ## CURRENTLY NOT CHECKING FOR TIP AND AFT!
    TILESIZE = map_instance["TileSize"]
    obstaclesDictionary = map_instance["obstacleDict"]
    cellOfPixel = (y//TILESIZE, x//TILESIZE, z//TILESIZE)

    return cellOfPixel in obstaclesDictionary

def arrived(current, map_instance, numberTree):
    """
    This function returns True if (x,y,z) is within the goal, False otherwise
    """

    # Get (x,y,z)
    x = current[0]
    y = current[1]
    z = current[2]

    # Compute goal area
    TILESIZE = map_instance["TileSize"]
    if numberTree == 1:
        arrivalx_min = TILESIZE * map_instance["goal_area"][1]
        arrivalx_max = arrivalx_min + TILESIZE
        arrivaly_min = TILESIZE * map_instance["goal_area"][0]
        arrivaly_max = arrivaly_min + TILESIZE
        arrivalz_min = TILESIZE * map_instance["goal_area"][2]
        arrivalz_max = arrivalz_min + TILESIZE
    else:
        arrivalx_min = TILESIZE * map_instance["start_area"][1]
        arrivalx_max = arrivalx_min + TILESIZE
        arrivaly_min = TILESIZE * map_instance["start_area"][0]
        arrivaly_max = arrivaly_min + TILESIZE
        arrivalz_min = TILESIZE * map_instance["start_area"][2]
        arrivalz_max = arrivalz_min + TILESIZE

    if (x<arrivalx_min or x>arrivalx_max) or (y<arrivaly_min or y>arrivaly_max)  or (z<arrivalz_min or z>arrivalz_max):
        return False
    
    if glbv.ARRIVED_PRIM == 0:
        print("DONE!")
        glbv.ARRIVED_PRIM = 1
        
    return True

def pointArrivedToGoal(current, goal_pixel):
    """
    This function returns True if (x,y,z) is within the goal, False otherwise
    """

    # Get (x,y,z)
    x = current[0]
    y = current[1]
    z = current[2]

    distance = np.sqrt((x - goal_pixel[0])**2 + (y - goal_pixel[1])**2 + (z - goal_pixel[2])**2)

    if distance > 0.2:
        return False
    
    if glbv.ARRIVED_PRIM == 0:
        print("DONE!")
        glbv.ARRIVED_PRIM = 1
        
    return True

def IsOutsideTheMap(x,y,z, map_instance):
    """
    This function checks if a pixel coordinate (x,y,z) is outside the map (True) or not (False) shrinked down by the radius of SAM = 0.095 meters
    """

    # Boundaries of the map
    xMin = 0 
    yMin = 0 
    zMin = 0 
    xMax = map_instance["x_max"] 
    yMax = map_instance["y_max"] 
    zMax = map_instance["z_max"] 
    radius = 0.095
    
    # Check if we are inside the map
    if (x > xMin + radius and y > yMin + radius and z > zMin + radius) and (x < xMax - radius and y < yMax - radius and z < zMax - radius):   # Inside the map
        return False
    
    return True

def compute_A_point_forward(state, distance=0.655):
    """
    Compute the point 7.405 meters forward along the vehicle's longitudinal axis
    """

    # Get current state
    x, y, z, q0, q1, q2, q3 = state[:7]

    # Create a rotation object from the quaternion
    rotation = R.from_quat([q1, q2, q3, q0])  # Note: scipy expects [x, y, z, w] order

    # Forward direction in body frame (longitudinal axis)
    forward_body = np.array([1, 0, 0])  # X-axis in body frame

    # Transform to world frame
    forward_world = rotation.apply(forward_body)
    forward_world /= np.linalg.norm(forward_world)

    # Compute new point
    new_point = np.array([x, y, z]) + distance * forward_world

    return tuple(new_point)

def compute_B_point_backward(state, distance=0.655):
    """
    Compute the point 7.405 meters forward along the vehicle's longitudinal axis
    """

    # Get current state
    x, y, z, q0, q1, q2, q3 = state[:7]

    # Create a rotation object from the quaternion
    rotation = R.from_quat([q1, q2, q3, q0])  # Note: scipy expects [x, y, z, w] order

    # Forward direction in body frame (longitudinal axis)
    forward_body = np.array([1, 0, 0])  # X-axis in body frame

    # Transform to world frame
    forward_world = rotation.apply(forward_body)
    forward_world /= np.linalg.norm(forward_world)

    # Compute new point
    new_point = np.array([x, y, z]) - distance * forward_world

    return tuple(new_point)

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

def calculate_angle_goalVector(state, vector, map_instance, numberTree, type = "normal"):
    """
    Compute the angle (in rad) between a vector and the goal vector from the current state
    """

    # Define current and goal positions
    x = state[0]
    y = state[1]
    z = state[2]
    '''
    match type:
        case "pointA":
                x_goal = map_instance["goal_pixel_pointA"][0]
                y_goal = map_instance["goal_pixel_pointA"][1]
                z_goal = map_instance["goal_pixel_pointA"][2]
        case _:
                x_goal = map_instance["goal_pixel"][0]
                y_goal = map_instance["goal_pixel"][1]
                z_goal = map_instance["goal_pixel"][2]
    '''
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

    # Define cvector
    vector_norm = np.linalg.norm(vector)
    if vector_norm != 0:
        vector /= vector_norm

    # Define the goal vector
    goal_vector = np.array([dx, dy, dz])
    goal_vector_norm = np.linalg.norm(goal_vector)
    if goal_vector_norm != 0:
        goal_vector /= goal_vector_norm

    # Boundary case
    if vector_norm == 0 or goal_vector_norm == 0:
        return 0 

    # Compute the angle 
    angle_between_vectors = calculate_angle_betweenVectors(vector, goal_vector) # both normalized

    # Return the value
    return angle_between_vectors    # in rad

def calculate_angle_betweenVectors(vector1, vector2):
    """
    Computes the angle (rad) between two vectors : [0, pi]
    """

    # Computing the norms
    vector1_norm = np.linalg.norm(vector1)
    vector2_norm = np.linalg.norm(vector2)

    # Extreme cases
    if vector1_norm == 0 or vector2_norm == 0:
        return 0
    
    # Computing the angle
    cos_theta = np.dot(vector1, vector2) / (vector1_norm * vector2_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_between_vectors = np.arccos(cos_theta)    # In rad [0, pi]

    # Return the value
    return angle_between_vectors