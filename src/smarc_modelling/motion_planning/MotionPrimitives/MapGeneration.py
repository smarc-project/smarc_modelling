'''
This script generates the map and its obstacle, boundaries, starting position and goal position.
The size of the grid is defined as TILESIZE
'''
import numpy as np
import random
from smarc_modelling.motion_planning.MotionPrimitives.GenerationTree import compute_current_forward_vector, calculate_angle_betweenVectors
from scipy.spatial.transform import Rotation as R
#import pandas as pd

def generationFirstMap():
    """Define random seed"""
    random.seed()

    """Define tiles"""
    F = 0   #free
    O = 1   #obstacle
    S = 2   #start
    G = 3   #goal

    """Boundaries"""
    do_you_want_boundaries = False

    """Probability of obstacle"""
    number_O = 0
    number_F = 100

    """Define tile colour"""
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    GREEN = (0,255,0)
    RED = (255,0,0)

    """Link tile to colour using DICTIONARY"""
    TileColour = {F : WHITE,
                O : BLACK,
                S : GREEN,
                G : RED
                }

    """Create the map"""
    # Generate obstacles (with probability)
    letters = [F]*number_F +[O]*number_O
    map1 = []
    # Vertical: 10m
    # Horizontal: 5m
    # Z: 3m

    TILESIZE = 0.5 #size of each cell in the grid
    numberVerticalTiles = int(10 // TILESIZE) #rows, y
    numberHorizontalTiles = int(5 // TILESIZE)  #columns, x
    number3DTiles = int(3 // TILESIZE) #z-axis
    # Minimum TileSize verifier
    if TILESIZE < 0.1875:
        print("ERROR: select a TileSize > 0.1875")
        exit(1)


    # Without boundaries
    for ii in range(number3DTiles):
        map1_horizontal = []    # map1 contains the list of maps in the horizontal plane. Each element is a 2d map treated as in MotionPrimitiveTest.
        for  jj in range(numberVerticalTiles):
            map1_horizontal.append(random.choices(letters, k=numberHorizontalTiles))  # add vertical boundaries 
        map1.append(map1_horizontal)


    ## Adding start to first layer and goal to top layer

    ## Nice maps:
    # (3,3) and (9,3)
    # (2,3) and (9,4)
    # (0,2) and (4,2)
    # (2,1) and (6,2)
    # (0,3) and (9,2)
    # (3,3) and (5,0)

    # Map with issues when optimizing:
    # (4, 2, 2) and (9, 9, 4)

        #start
    startrCell = random.randrange(numberVerticalTiles-4) + 2  #random CELL
    startcCell = random.randrange(numberHorizontalTiles-4) + 2 
    startzCell = random.randrange(number3DTiles)                                           
    map1[startzCell][startrCell][startcCell] = 2

        #goal center cell
    goalrCell = random.randrange(numberVerticalTiles-4) + 2     #random CELL
    goalcCell = random.randrange(numberHorizontalTiles - 4) + 2    #random CELL
    goalzCell = random.randrange(number3DTiles)

    # Change the goal if goal==start position
    while goalrCell == startrCell and goalcCell == startcCell and goalzCell == startzCell:
        goalrCell = random.randrange(numberVerticalTiles - 4) + 2    #random CELL
        goalcCell = random.randrange(numberHorizontalTiles - 4) + 2    #random CELL
        goalzCell = random.randrange(number3DTiles)

    map1[goalzCell][goalrCell][goalcCell] = 3

    """Create the map-size"""
    mapWidth = numberHorizontalTiles * TILESIZE #in pixel metric
    mapHeight = numberVerticalTiles * TILESIZE  #in pixel metric
    map3DSize = number3DTiles * TILESIZE        #in pixel metric

    """Create the dictionaries"""
    ## Obstacle dictionary
    obstaclesDictionary = {(r, c, z) for z in range(number3DTiles) for r in range(numberVerticalTiles) for c in range(numberHorizontalTiles) if map1[z][r][c] == 1}

    ## Starting pixel (center of the starting cell)
    startingPixel = (startcCell * TILESIZE + 0.5 * TILESIZE, startrCell * TILESIZE + 0.5 * TILESIZE, startzCell * TILESIZE + 0.5 * TILESIZE)    #ATTENTION! is (x,y,z), not row, column and z

    ## Goal pixel (center of the goal cell)
    arrivalPixel = (goalcCell * TILESIZE + 0.5 * TILESIZE, goalrCell * TILESIZE + 0.5 * TILESIZE, goalzCell * TILESIZE + 0.5 * TILESIZE)    #ATTENTION! is (x,y,z), not row, column and z

    # SAM initial state 
    eta0 = np.zeros(7)
    eta0[0] = startingPixel[0]
    eta0[1] = startingPixel[1]
    eta0[2] = startingPixel[2]
    initial_yaw = np.deg2rad(random.randrange(-180, 180, 90))   # in deg
    initial_pitch = np.deg2rad(0) # in deg
    initial_roll = np.deg2rad(0)  # in deg 
    r = R.from_euler('zyx', [initial_yaw, initial_pitch, initial_roll])
    q0 = r.as_quat()
    eta0[3] = q0[3]
    eta0[4:7] = q0[0:3]
    nu0 = np.zeros(6)   # Zero initial velocities
    u0 = np.zeros(6)    #The initial control inputs for SAM
    u0[0] = 50          #Vbs
    u0[1] = 50          #lcg
    x0 = np.concatenate([eta0, nu0, u0])
    
    # SAM final state
    finalState = x0.copy()
    finalState[0:3] = arrivalPixel[0:3]
    final_yaw = np.deg2rad(random.randrange(-180, 180, 90))   # in deg
    final_pitch = np.deg2rad(0) # in deg
    final_roll = np.deg2rad(0)  # in deg 
    r = R.from_euler('zyx', [final_yaw, final_pitch, final_roll])
    q = r.as_quat()
    finalState[3] = q[3]
    finalState[4:7] = q[0:3]

    # Create the map instance to pass to other scripts
    map_instance = {
        "x_max": mapWidth,
        "y_max": mapHeight,
        "z_max": map3DSize,
        "x_min": 0,
        "y_min": 0,
        "z_min": 0,
        "obstacleDict": obstaclesDictionary,
        "start_pos": startingPixel, #(x,y,z)
        "start_area": (startrCell, startcCell, startzCell),
        "goal_area": (goalrCell, goalcCell, goalzCell),    #(CELLy, CELLx, CELLz)  <---This is the one working for sure
        "goal_pixel": (arrivalPixel[0], arrivalPixel[1], arrivalPixel[2]),   #(x,y,z)  <---This is the one working for sure
        "final_state": finalState,
        "TileSize": TILESIZE,
        "initial_state": x0
    }

    # Saving map_instance
    #map_resume = []
    #map_resume.append((map_instance["start_area"], map_instance["goal_area"], np.rad2deg(final_yaw)))
    #df = pd.DataFrame(map_resume, columns=["start_area", "goal_area", "final_yaw"])
    #df.to_csv("last_map_instance.csv", index=False)

    return map_instance

def generateMapInstance(start_state, goal_state, map_boundaries, map_res):
    """
    Given a start state and a goal state, this function generates a map_instance dictionary compatible with the motion primitives algorithm.
    """

    # Define a TileSize:
    TILESIZE = map_res  #meters

    # Get the start and goal positions
    startingPixel = (start_state[0], start_state[1], start_state[2])
    arrivalPixel = (goal_state[0], goal_state[1], goal_state[2])

    # Get the start area
    lengthX = np.abs(map_boundaries[0]) + np.abs(map_boundaries[3])
    lengthY = np.abs(map_boundaries[1]) + np.abs(map_boundaries[4])
    lengthZ = np.abs(map_boundaries[2]) + np.abs(map_boundaries[5])

    if startingPixel[1] < 0:
        startrCell = int(np.abs(map_boundaries[4]) / TILESIZE - 1) - np.abs(startingPixel[1]) // TILESIZE
    else: 
        startrCell = int(np.abs(map_boundaries[4]) / TILESIZE) + startingPixel[1] // TILESIZE
    
    if startingPixel[0] < 0:
        startcCell = int(np.abs(map_boundaries[3]) / TILESIZE - 1) - np.abs(startingPixel[0]) // TILESIZE
    else:
        startcCell = int(np.abs(map_boundaries[3]) / TILESIZE) + startingPixel[0] // TILESIZE

    if startingPixel[2] < 0:
        startzCell = int(np.abs(map_boundaries[5]) / TILESIZE - 1) - np.abs(startingPixel[2]) // TILESIZE
    else:
        startzCell = int(np.abs(map_boundaries[5]) / TILESIZE) + startingPixel[2] // TILESIZE

    # Get the goal area
    if arrivalPixel[1] < 0:
        goalrCell = int(np.abs(map_boundaries[4]) / TILESIZE - 1) - np.abs(arrivalPixel[1]) // TILESIZE
    else: 
        goalrCell = int(np.abs(map_boundaries[4]) / TILESIZE) + arrivalPixel[1] // TILESIZE
    
    if arrivalPixel[0] < 0:
        goalcCell = int(np.abs(map_boundaries[3]) / TILESIZE - 1) - np.abs(arrivalPixel[0]) // TILESIZE
    else:
        goalcCell = int(np.abs(map_boundaries[3]) / TILESIZE) + arrivalPixel[0] // TILESIZE
    
    if arrivalPixel[2] < 0:
        goalzCell = int(np.abs(map_boundaries[5]) / TILESIZE - 1) - np.abs(arrivalPixel[2]) // TILESIZE
    else:
        goalzCell = int(np.abs(map_boundaries[5]) / TILESIZE) + arrivalPixel[2] // TILESIZE

    # Create the map instance to pass to other scripts
    map_instance = {
        "x_max": map_boundaries[0], #meters
        "y_max": map_boundaries[1],    #meters
        "z_max": map_boundaries[2], #meters
        "x_min": map_boundaries[3],
        "y_min": map_boundaries[4],
        "z_min": map_boundaries[5],
        "obstacleDict": [], #not important for the tank
        "start_pos": startingPixel, #(x,y,z)
        "start_area": (startrCell, startcCell, startzCell),
        "goal_area": (goalrCell, goalcCell, goalzCell),    #(CELLy, CELLx, CELLz)  <---This is the one working for sure
        "goal_pixel": arrivalPixel,   #(x,y,z) 
        "final_state": goal_state,
        "TileSize": TILESIZE,
        "initial_state": start_state
    }

    return map_instance

def evaluateComplexityMap(map_instance):
    start_position = map_instance["start_pos"]
    goal_position = map_instance["goal_pixel"]
    start_quaternion = map_instance["initial_state"][3:7]
    goal_quaternion = map_instance["final_state"][3:7]
    start_state = np.concatenate([start_position, start_quaternion])
    goal_state = np.concatenate([goal_position, goal_quaternion])

    ## Compute the angle between the forward vectors and start_goal_vector (complexity 0)
    forward_start = compute_current_forward_vector(start_state)
    forward_goal = compute_current_forward_vector(goal_state)
    angle = calculate_angle_betweenVectors(forward_start, forward_goal)
    start_goal_vector = (goal_state[0]-start_state[0], goal_state[1]-start_state[1], goal_state[2]-start_state[2])
    angle2 = calculate_angle_betweenVectors(start_goal_vector, forward_start)
    
    #if np.abs(np.rad2deg(angle)) < 7 and (np.abs(np.rad2deg(angle2)) < 7 or np.abs(np.rad2deg(angle2-np.pi)) < 7):
    #    return 0
    
    ## Distance between goal/start (+1)
    #linear_distance = np.sqrt((start_position[0] - goal_position[0])**2 + (start_position[1] - goal_position[1])**2 + (start_position[2] - goal_position[2])**2)

    ## Difference in z positions goal/start (+5)
    difference_z = np.abs(goal_position[2] - start_position[2])
    xy_distance = np.sqrt((start_position[0] - goal_position[0])**2 + (start_position[1] - goal_position[1])**2)

    # Complexity 3
    if difference_z > 0 and xy_distance > 1:    # xyz 
        return 3
    elif difference_z > 0:  # only z
        return 2
    elif xy_distance < 3 and np.abs(np.rad2deg(angle)) < 7 and (np.abs(np.rad2deg(angle2)) < 7 or np.abs(np.rad2deg(angle2-np.pi)) < 7):    # single tree
        return 0
    else:                   # xy
        return 1
    