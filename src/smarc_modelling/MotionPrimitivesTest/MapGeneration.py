'''
This script generates the map and its obstacle, boundaries, starting position and goal position.
The size of the grid is defined as TILESIZE
'''

import pygame, sys, numpy as np
import random

"""Define random seed"""
random.seed()

"""Define tiles"""
F = 0   #free
O = 1   #obstacle
S = 2   #start
G = 3   #goal

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
## Generating obstacles
    #probability of obstacles
letters = [F,F,O,F,F, F, F, F, F, F]
    #generation of obstacles
map1 = []
numberVerticalTiles = 20 #rows
numberHorizontalTiles = 30  #columns
'''
# With boundaries
for jj in range(numberVerticalTiles):
    if jj == 0 or jj == numberVerticalTiles-1:   #add horizontal boundaries
        map1.append([1]*numberHorizontalTiles)
    else:
        map1.append([1] + random.choices(letters, k=numberHorizontalTiles-2)+ [1]) #add vertical boundaries
'''

# Without boundaries
for jj in range(numberVerticalTiles):
    map1.append(random.choices(letters, k=numberHorizontalTiles)) #add vertical boundaries

## Adding boundaries
#horizontalBorder = [1] * numberHorizontalTiles
#map1[0] = horizontalBorder
#map1[numberVerticalTiles-1] = horizontalBorder
    #vertical
#for ii in range(numberVerticalTiles):
#    map1[ii][0] = 1
#    map1[ii][numberHorizontalTiles-1] = 1

## Adding start and goal
    #start
randomPositionStart = random.randrange(numberVerticalTiles-2)+1   #random CELL 
map1[randomPositionStart][1] = 2
    #goal
randomPositionGoal = random.randrange(numberVerticalTiles-2)+1    #random CELL
map1[randomPositionGoal][numberHorizontalTiles-2] = 3

"""Create the map-size"""
TILESIZE = 30 #size of each cell in the grid
mapWidth = numberHorizontalTiles * TILESIZE #in pixel metric
mapHeight = numberVerticalTiles * TILESIZE  #in pixel metric

"""Create the dictionaries"""
## Obstacle dictionary
obstaclesDictionary = {(x,y) for y in range(numberVerticalTiles) for x in range(numberHorizontalTiles) if map1[y][x] == 1}
    #How to check --> if (x,y) in obstaclesDictionary == true...

## Starting pixel (center of the starting cell)
startingPixel = (TILESIZE + 0.5 * TILESIZE, randomPositionStart * TILESIZE + 0.5 * TILESIZE)

## Goal dictionary
goalDictionary = {(numberHorizontalTiles-2, randomPositionGoal): "cellGoal"}
goalCellx, goalCelly = next(iter(goalDictionary))   #Unpacking the goal cell from dictionary
arrivalPixel = (TILESIZE * goalCellx + 0.5 * TILESIZE, randomPositionGoal * TILESIZE + 0.5 * TILESIZE )

# Create the map instance
map = {
    "x_max": mapWidth,
    "y_max": mapHeight,
    "z_max": 0,
    "obstacleDict": obstaclesDictionary,
    "start_pos": startingPixel, #(x,y)
    "goal_area": (goalCellx, goalCelly),    #(CELLx, CELLy)
    "goal_pixel": (arrivalPixel[0], arrivalPixel[1])
}



'''
"""Create display"""
pygame.init()
DISPLAY = pygame.display.set_mode((mapWidth, mapHeight))

"Draw map to Display"
#ROWS
for row in range(numberVerticalTiles):
    #COLUMNS
    for col in range(numberHorizontalTiles):
        #DRAW TILE
        pygame.draw.rect(DISPLAY, TileColour[map1[row][col]],(col*TILESIZE, row*TILESIZE, TILESIZE, TILESIZE))

"""Basic user interface"""
while True:
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
    
    "Update Display"
    pygame.display.update()
'''

