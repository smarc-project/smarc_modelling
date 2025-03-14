import sys
sys.path.append('~/Desktop/smarc_modelling-master')
from smarc_modelling.MotionPrimitivesTest.MapGeneration import obstaclesDictionary, TILESIZE
'''
--> How to check if a pixel position (x,y) is within an obstacle: 
IsWithinObstacle(x,y)
'''
def IsWithinObstacle(x,y):
    ##This function checks if a PIXEL (x,y) is within an obstacle defined by obstacleDisctionary
    ##It returns True if (x,y) collides with an obstacle and False otherwise

    cellOfPixel = (x//TILESIZE, y//TILESIZE)
    for key in obstaclesDictionary:
        if key[0] == cellOfPixel[0] and key[1] == cellOfPixel[1]:
            return True
    return False
