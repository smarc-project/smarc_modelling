import sys
import numpy as np
import smarc_modelling.MotionPrimitives.GlobalVariables_MotionPrimitives as glbv
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
        maxFinalVelocity = 0.5

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

def calculate_angle_goalVector(state, vector, map_instance):
    """
    Compute the angle (in rad) between a vector and the goal vector from the current state
    """

    # Define current and goal positions
    x = state[0]
    y = state[1]
    z = state[2]
    x_goal = map_instance["goal_pixel"][0]
    y_goal = map_instance["goal_pixel"][1]
    z_goal = map_instance["goal_pixel"][2]

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

def draw_torpedo(ax, vertex, colorr, length=1.5, radius=0.095, resolution=20):
    """
    Draws a torpedo-like shape (cylinder) and a black actuator at the back (disk) at (x, y, z) with orientation from quaternion.
    """

    # Find the parameters
    x, y, z, q0, q1, q2, q3 = vertex[:7]

    # Create cylinder (torpedo body)
    theta = np.linspace(0, 2 * np.pi, resolution)
    x_cyl = np.linspace(-0.5, 0.5, resolution) * length  # adjusting length
    theta, x_cyl = np.meshgrid(theta, x_cyl)
    y_cyl = radius * np.cos(theta)
    z_cyl = radius * np.sin(theta)
    
    # Create hemispherical caps
    r_disk = np.linspace(0, radius, resolution)  # Radial distances
    theta_disk = np.linspace(0, 2 * np.pi, resolution)  # Angles
    r_disk, theta_disk = np.meshgrid(r_disk, theta_disk)
    x_cap_rear = np.full_like(r_disk, -0.5 * length)  # Fixed x position (rear end of the torpedo)
    y_cap_rear = r_disk * np.cos(theta_disk)
    z_cap_rear = r_disk * np.sin(theta_disk)

    # Convert quaternion to rotation matrix
    r = R.from_quat([q1, q2, q3, q0]) 
    rotation_matrix = r.as_matrix()
    
    # Apply rotation
    def transform_points(x, y, z):
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()])
        rotated_points = rotation_matrix @ points  # Matrix multiplication
        return rotated_points[0].reshape(x.shape), rotated_points[1].reshape(y.shape), rotated_points[2].reshape(z.shape)
    x_cyl, y_cyl, z_cyl = transform_points(x_cyl, y_cyl, z_cyl)
    x_cap_rear, y_cap_rear, z_cap_rear = transform_points(x_cap_rear, y_cap_rear, z_cap_rear)
    
    # Apply translation
    x_cyl += x
    y_cyl += y
    z_cyl += z
    x_cap_rear += x 
    y_cap_rear += y
    z_cap_rear += z
    
    # Plot spheres
    plotSpheres = False
    if plotSpheres:
        pointA = compute_A_point_forward(vertex)
        pointB = compute_B_point_backward(vertex)
        radius = 0.095
        u = np.linspace(0, 2 * np.pi, 30)  # azimuthal angle
        v = np.linspace(0, np.pi, 30)      # polar angle

        # Convert spherical to Cartesian coordinates
        XA = radius * np.outer(np.cos(u), np.sin(v)) + pointA[0]
        YA = radius * np.outer(np.sin(u), np.sin(v)) + pointA[1]
        ZA = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pointA[2]
        XB = radius * np.outer(np.cos(u), np.sin(v)) + pointB[0]
        YB = radius * np.outer(np.sin(u), np.sin(v)) + pointB[1]
        ZB = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pointB[2]

        # Plot the sphere
        ax.plot_surface(XA, YA, ZA, color='k', alpha=0.6, edgecolor='k')
        ax.plot_surface(XB, YB, ZB, color='k', alpha=0.6, edgecolor='k')
    
    # Plot surfaces (cylinder and cap)
    ax.plot_surface(x_cyl, y_cyl, z_cyl, color='y', alpha=colorr)
    ax.plot_surface(x_cap_rear, y_cap_rear, z_cap_rear, color='k', alpha=colorr)