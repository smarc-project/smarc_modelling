import numpy as np
import multiprocessing
import math

def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points
    """

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def interpolate(args):
    """
    Check if the distance between two points exceeds maxDistance
    """

    # Check how many new points we need
    trajectory, max_distance, index = args
    distance = euclidean_distance(trajectory[index], trajectory[index + 1])
    numberNewPoints = int(distance // max_distance)
    if distance % max_distance == 0 :
        numberNewPoints -= 1

    finalPointList =  []
    #newPoint = trajectory[index]
    newPoint = 1
    u = trajectory[index+1][13:19]

    for i in range(numberNewPoints + 1):
        finalPointList.append(newPoint)
        newPoint = 2
    finalPointList.append(1)
    return (distance, numberNewPoints, finalPointList)

def increaseResolutionTrajectory(trajectory, maximumDistance):
    """
    Analyzes a trajectory and marks where interpolation is needed
    """

    if len(trajectory) < 2:
        return []
    
    with multiprocessing.Pool() as pool:
        indices = range(len(trajectory) - 1)
        results = pool.map(interpolate, [(trajectory, maximumDistance, i) for i in indices])
    
    return results
