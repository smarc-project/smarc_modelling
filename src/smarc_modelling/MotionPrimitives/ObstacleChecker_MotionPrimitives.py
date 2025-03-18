import sys
import numpy as np
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

def arrived(current, map_instance):
    """
    This function returns True if (x,y,z) is within the goal, False otherwise
    """

    # Get (x,y,z)
    x = current[0]
    y = current[1]
    z = current[2]
    
    # Constraint on final velocity
    '''
    finalVelocityConstraint = False
    if finalVelocityConstraint:
        q0, q1, q2, q3, vx, vy, vz = current[3:10]
        currentVelocityVector = body_to_global_velocity((q0,q1,q2,q3), [vx, vy, vz])
        currentVelocityNorm = np.linalg.norm(currentVelocityVector)
        maxFinalVelocity = 1
        if currentVelocityNorm > maxFinalVelocity:
            return False
    '''

    # Compute goal area
    TILESIZE = map_instance["TileSize"]
    arrivalx_min = TILESIZE * map_instance["goal_area"][1]
    arrivalx_max = arrivalx_min + TILESIZE
    arrivaly_min = TILESIZE * map_instance["goal_area"][0]
    arrivaly_max = arrivaly_min + TILESIZE
    arrivalz_min = TILESIZE * map_instance["goal_area"][2]
    arrivalz_max = arrivalz_min + TILESIZE

    if (x<arrivalx_min or x>arrivalx_max) or (y<arrivaly_min or y>arrivaly_max)  or (z<arrivalz_min or z>arrivalz_max):
        return False
    
    print("DONE!")
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

def IsFinedInRestricted(current, map_instance):
    """
    SAM is fined if its current velocity norm overcomes the speed limit in the area
    """

    # Get (x,y,z)
    x, y, z = current[0:3]

    # Get the restricted area boundaries
    (restr_x_min, restr_y_min, restr_z_min), (restr_x_max, restr_y_max, restr_z_max) = map_instance["restricted_area"]

    # Check if we are inside the restricted area
    if (x>=restr_x_min and x<=restr_x_max) and (y>=restr_y_min and y<=restr_y_max) and (z>=restr_z_min and z<=restr_z_max):
        q0, q1, q2, q3, vx, vy, vz = current[3:10]
        currentVelocityVector = body_to_global_velocity((q0,q1,q2,q3), [vx, vy, vz])
        currentVelocityNorm = np.linalg.norm(currentVelocityVector)
        maxFinalVelocity = 1.5

        # Check if our velocity exceeds the limit
        if(currentVelocityNorm > maxFinalVelocity):
            return True
        
    return False

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