### To Do List ###
# 0. v/orientation in InputPair (GenTree), getNeigh (GenTree), calculate_f (GenTree) 
# 3. Find right decelleration

# angles: get_neigh, curvePrimitives
import numpy as np
from smarc_modelling.motion_planning.MotionPrimitives.ObstacleChecker import compute_A_point_forward, compute_B_point_backward
from smarc_modelling.vehicles.SAM import SAM 
import matplotlib
import matplotlib.pyplot as plt
import csv
from scipy.spatial.transform import Rotation as R
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have 

def plot_map(map_data, typePlot):
    """
    This function creates the map to be shown. It colors the cells, create the tank and set the initial view with the axis limit.
    """
    #line
    # Extract map properties
    map_x_max = map_data["x_max"]   #number of column-wise
    map_y_max = map_data["y_max"]   #number of row-wise
    map_z_max = map_data["z_max"]
    obstacles = map_data["obstacleDict"]    #in the grid (row, column and z)
    start_pos = map_data["start_pos"]   #(x,y,z)
    goal_pixel = map_data["goal_pixel"] #(x,y,z)
    goal_area = map_data["goal_area"]
    goal_area_front = map_data["goal_area_front"]
    TILESIZE = map_data["TileSize"]
    (restr_x_min, restr_y_min, restr_z_min), (restr_x_max, restr_y_max, restr_z_max) = map_data["restricted_area"]

    # 3D plot setup
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot map as 3D surface 
    x_grid = np.arange(0, map_x_max, TILESIZE)
    y_grid = np.arange(0, map_y_max, TILESIZE)
    z_grid = np.arange(0, map_z_max, TILESIZE)
    xx, yy , zz = np.meshgrid(x_grid, y_grid, z_grid)   # Shape: (numberVertical, numberHorizontal, number3D) == (y,x,z)

    # Create color grid for map visualization
    color_grid = np.full_like(xx, "white", dtype=object)    #(rows, columns, z)
    for obs in obstacles:
        color_grid[obs[0], obs[1], obs[2]] = "black"
    color_grid[int(start_pos[1] // TILESIZE), int(start_pos[0] // TILESIZE), int(start_pos[2] // TILESIZE)] = "green"
    color_grid[int(goal_pixel[1] // TILESIZE), int(goal_pixel[0] // TILESIZE), int(goal_pixel[2] // TILESIZE)] = "red"
    #color_grid[int(goal_area_front[0]), int(goal_area_front[1]), int(goal_area_front[2])] = "red"
    '''
    if map_data["where"] == "top":
        color_grid[int(goal_area[0]-1), int(goal_area[1]), int(goal_area[2])] = "red"
        color_grid[int(goal_area[0]-2), int(goal_area[1]), int(goal_area[2])] = "red"
        color_grid[int(goal_area[0]+2), int(goal_area[1]), int(goal_area[2])] = "red"
        color_grid[int(goal_area[0]+1), int(goal_area[1]), int(goal_area[2])] = "red"

    else:
        color_grid[int(goal_area[0]), int(goal_area[1]-1), int(goal_area[2])] = "red"
        color_grid[int(goal_area[0]), int(goal_area[1]-2), int(goal_area[2])] = "red"
        color_grid[int(goal_area[0]), int(goal_area[1]+1), int(goal_area[2])] = "red"
        color_grid[int(goal_area[0]), int(goal_area[1]+2), int(goal_area[2])] = "red"
    '''
    
    plotRestrictedArea = False
    if plotRestrictedArea:
        # Transform restricted area from boundaries to cells
        restr_c_min_idx = int(restr_x_min // TILESIZE)
        restr_c_max_idx = int(restr_x_max // TILESIZE)
        restr_r_min_idx = int(restr_y_min // TILESIZE)
        restr_r_max_idx = int(restr_y_max // TILESIZE)
        restr_z_min_idx = int(restr_z_min // TILESIZE)
        restr_z_max_idx = int(restr_z_max // TILESIZE)

        # Define restricted area in color_grid
        for k in np.arange(restr_z_min_idx, restr_z_max_idx):  # z-axis
            for i in np.arange(restr_r_min_idx, restr_r_max_idx):  # row-axis (y)
                for j in np.arange(restr_c_min_idx, restr_c_max_idx):  # column-axis (x)
                    color_grid[i, j, k] = "cyan"  # Mark as restricted

    # Plot colored tiles from color_grid
    for k in range(color_grid.shape[2]):  # z-axis
        for i in range(color_grid.shape[0]):  # row-axis
            for j in range(color_grid.shape[1]):  # column-axis
                if color_grid[i,j,k] == "black" or color_grid[i,j,k] == "green" or color_grid[i,j,k] == "red":
                    ax.bar3d(
                        xx[i, j, k], 
                        yy[i, j, k],
                        zz[i, j, k],
                        TILESIZE, TILESIZE, TILESIZE,
                        color=color_grid[i, j, k],
                        edgecolor='gray',
                        alpha=0.9
                    )
                elif color_grid[i,j,k] == "cyan":
                    ax.bar3d(
                        xx[i, j, k], 
                        yy[i, j, k],
                        zz[i, j, k],
                        TILESIZE, TILESIZE, TILESIZE,
                        color="red",
                        edgecolor="none",
                        alpha=0.01
                    )

    # Plot start and goal points
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='green', marker='o', s=100, label="Start")
    ax.scatter(goal_pixel[0], goal_pixel[1], goal_pixel[2], c='red', marker='x', s=100, label="End")

    # Axis labels
    ax.set_xlabel('X / East')
    ax.set_ylabel('Y / North')
    ax.set_zlabel('-Z / Down')
    ax.set_title('Trajectory with Map')
    ax.legend()

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

    # return the plot setup
    return (ax, plt, fig)

def update(frame, ax, plt, trajectory):

    # Get the current state
    vertex = trajectory[frame]
    colorr = frame / len(trajectory)  # Normalize color based on frame

    # Draw torpedo
    draw_torpedo(ax, vertex, colorr)

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
            except ValueError:
                print(f"Warning: Could not convert row to numbers: {row}. Skipping.")
                continue
            trajectory.append(state)
    return trajectory

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
