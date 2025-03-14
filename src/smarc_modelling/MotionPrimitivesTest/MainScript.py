'''
This script is the main. It draws the obstacle map, and primitives on the starting position
'''
import heapq
import math
import sys
sys.path.append('~/Desktop/smarc_modelling-master')
from smarc_modelling.MotionPrimitivesTest.MapGeneration import *
from smarc_modelling.MotionPrimitivesTest.ObstacleChecker import *
from smarc_modelling.MotionPrimitivesTest.MotionPrimitives import SAM_PRIMITIVES
from smarc_modelling.MotionPrimitivesTest.GenerationTree import a_star_search

"""Initialize variables"""
max_input = 7   #In degrees
step_input = 1
# The initial state
eta0 = np.zeros(7)
eta0[0] = startingPixel[0]
eta0[1] = startingPixel[1]
eta0[2] = 0
eta0[3] = 1.0       # Initial quaternion (no rotation) 
nu0 = np.zeros(6)   # Zero initial velocities
u0 = np.zeros(6)    #The initial control inputs for SAM
u0[0] = 50          #Vbs
u0[1] = 50          #lcg
x0 = np.concatenate([eta0, nu0, u0])
# The initial lists of nodes and parents
grid = []   #It will be also the open set!
parents_list = {}
# Add start and goal to the list and dictionary
start = (startingPixel[0], startingPixel[1])
goal = (arrivalPixel[0], arrivalPixel[1])
heapq.heappush(grid, (0, start))
parents_list[start] = (0, None) #save cost and parent

"""Create display"""
pygame.init()
DISPLAY = pygame.display.set_mode((mapWidth, mapHeight))
simulator = SAM_PRIMITIVES()

"Draw map to Display"
#ROWS
for row in range(numberVerticalTiles):
    #COLUMNS
    for col in range(numberHorizontalTiles):
        #DRAW TILE
        pygame.draw.rect(DISPLAY, TileColour[map1[row][col]],(col*TILESIZE, row*TILESIZE, TILESIZE, TILESIZE))

"""Create a path"""
path = a_star_search(x0, DISPLAY)
#print(path)

"""Basic user interface"""
while True:

    #close the pygame window if I want to
    for event in pygame.event.get():
        """Quit (when "x" in the top corner is pressed)"""
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    "Draw map to Display"
    #ROWS
    for row in range(numberVerticalTiles):
        #COLUMNS
        for col in range(numberHorizontalTiles):
            #DRAW TILE
            pygame.draw.rect(DISPLAY, TileColour[map1[row][col]],(col*TILESIZE, row*TILESIZE, TILESIZE, TILESIZE))

    
    # Draw the path
    for index in range(len(path)-1):
        pygame.draw.line(DISPLAY, (255,0,0), (path[index][0], path[index][1]), (path[index + 1][0], path[index + 1][1]), 2)
    

    '''
    "Draw random primitives"
    for input_u in np.arange(-max_input,max_input,step_input):
        #get the states if I apply such input
        data = simulator.curvePrimitives(x0, input_u) #Replace with simulator.curvePrimitives_justToShow for plotting
        x_val = data[0,:]
        y_val = data[1,:]
        z_val = data[2,:]
        for ii in range(1,len(x_val)):
            if not IsWithinObstacle(x_val[ii], y_val[ii]):
                pygame.draw.line(DISPLAY, (0,0,255), (x_val[ii-1], y_val[ii-1]), (x_val[ii], y_val[ii]), 2)
            else:
                break
    '''

    "Update Display"
    pygame.display.update()

