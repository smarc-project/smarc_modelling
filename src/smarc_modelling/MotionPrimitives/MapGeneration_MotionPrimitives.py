'''
This script generates the map and its obstacle, boundaries, starting position and goal position.
The size of the grid is defined as TILESIZE
'''
import numpy as np
import random

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

    TILESIZE = 0.2 #size of each cell in the grid
    numberVerticalTiles = int(10 // 0.2) #rows, y
    numberHorizontalTiles = int(5 // 0.2)  #columns, x
    number3DTiles = int(3 // 0.2) #z-axis
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
    startrCell = random.randrange(numberVerticalTiles)  #random CELL
    startcCell = random.randrange(numberHorizontalTiles-2)+1 
    startrCell = 4      # row = meters*5
    startcCell = 5
    startzCell = 1
    map1[startzCell][startrCell][startcCell] = 2

        #goal center cell
    goalrCell = random.randrange(numberVerticalTiles)    #random CELL
    goalcCell = random.randrange(numberHorizontalTiles)    #random CELL
    goalrCell = 25      
    goalcCell = 20
    goalzCell = 10
    map1[goalzCell][goalrCell][goalcCell] = 3

        #orientation wrt goalCell --> the cell minimum
    where = "top"

    """Create the map-size"""
    mapWidth = numberHorizontalTiles * TILESIZE #in pixel metric
    mapHeight = numberVerticalTiles * TILESIZE  #in pixel metric
    map3DSize = number3DTiles * TILESIZE        #in pixel metric

    """Create the dictionaries"""
    ## Obstacle dictionary
    obstaclesDictionary = {(r, c, z) for z in range(number3DTiles) for r in range(numberVerticalTiles) for c in range(numberHorizontalTiles) if map1[z][r][c] == 1}
    #for key in obstaclesDictionary:
        #print("OBSTDICT")
        #print(key)
        #How to check --> if (y,x,z) in obstaclesDictionary == true...

    ## Starting pixel (center of the starting cell)
    startingPixel = (startcCell * TILESIZE + 0.5 * TILESIZE, startrCell * TILESIZE + 0.5 * TILESIZE, startzCell * TILESIZE + 0.5 * TILESIZE)    #ATTENTION! is (x,y,z), not row, column and z

    ## Goal pixel (center of the goal cell)
    arrivalPixel = (goalcCell * TILESIZE + 0.5 * TILESIZE, goalrCell * TILESIZE + 0.5 * TILESIZE, goalzCell * TILESIZE + 0.5 * TILESIZE)    #ATTENTION! is (x,y,z), not row, column and z

    ## Restricted area (where the velocity will be constrained)
    bound = 1
    restr_x_min = max(0, arrivalPixel[0] - bound)
    restr_x_max = min(mapWidth, arrivalPixel[0] + bound)
    restr_y_min = max(0, arrivalPixel[1] - bound)
    restr_y_max = min(mapHeight, arrivalPixel[1] + bound)
    restr_z_min = max(0, arrivalPixel[2] - bound)
    restr_z_max = min(map3DSize, arrivalPixel[2] + bound)

    ## Goal pixel for front of SAM
    # Case: with respect to the goal pixel for the CG. Looking from the bottom of the tank.    <---columns
    #                                                                                             |
    #                                                                                             |
    #                                                                                             v rows
    match where:
        case "top":
            goalAreaFront = (goalrCell - 1, goalcCell, goalzCell)
        case _:
            goalAreaFront = (goalrCell, goalcCell + 1, goalzCell) # right
    
    # Create the map instance to pass to other scripts
    map_instance = {
        "x_max": mapWidth,
        "y_max": mapHeight,
        "z_max": map3DSize,
        "obstacleDict": obstaclesDictionary,
        "start_pos": startingPixel, #(x,y,z)
        "goal_area": (goalrCell, goalcCell, goalzCell),    #(CELLy, CELLx, CELLz)
        "goal_area_front": goalAreaFront,  #(CELLy, CELLx, CELLz)
        "restricted_area": [(restr_x_min, restr_y_min, restr_z_min), (restr_x_max, restr_y_max, restr_z_max)],  # [(minimumCoordinates), (maximumCoordinates)], list of tuples
        "goal_pixel": (arrivalPixel[0], arrivalPixel[1], arrivalPixel[2]),   #(x,y,z)
        "TileSize": TILESIZE,
        "where": where
    }

    return map_instance
