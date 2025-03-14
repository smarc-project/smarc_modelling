import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('~/Desktop/smarc_modelling-master')
from mpl_toolkits.mplot3d import Axes3D
from smarc_modelling.MotionPrimitivesTest.ObstacleChecker import IsWithinObstacle

'''
This script defines the Dubin primitives.
It is called by the MainScript.py to draw the primitives. If you want to draw the primitives locally, just 
edit this script changing curvePrimitives() with curvePrimitives_justToShow()

The __main__ plots the local primitives
'''

class DubinSimulator:
    def __init__(self):
        self.dt = 0.5
        self.t_max = 5
        self.v = 10

    def curvePrimitives_singleStep(self, x0=0, y0=0, theta0=0, u=1):
        #dynamical model with forward Euler, it returns a SINGLE step
        x1 = x0 + self.dt * self.v * np.cos(theta0)
        y1 = y0 + self.dt * self.v * np.sin(theta0)
        z1 = 0
        theta1 = theta0 + u

        #check if I'm within in an obstacle
        if IsWithinObstacle(x1,y1):
            return x1, y1, z1, theta1, True
        else:
            return x1, y1, z1, theta1, False
    
    def curvePrimitives(self, x0, y0, theta0, u):
        #It returns the sequence of steps within a single input
        t = np.arange(0, self.t_max, self.dt)
        lengthT = len(t)
        x = [x0]
        y = [y0]
        z = [0]
        theta = [theta0]

        for _ in range(lengthT):
            x_next, y_next, z_next, theta_next, isInObstacle = self.curvePrimitives_singleStep(x[-1], y[-1], theta[-1], u)
            if not isInObstacle:
                x.append(x_next)
                y.append(y_next)
                z.append(z_next)
                theta.append(theta_next)
            else:
                x = [x0]
                y = [y0]
                z = [0]
                theta = [theta0]
                return x,y,z,theta
    
        return x,y,z,theta
    
    def curvePrimitives_singleStep_justToShow(self, x0=0, y0=0, theta0=0, u=1):
        #dynamical model with forward Euler, it returns a SINGLE step
        x1 = x0 + self.dt * self.v * np.cos(theta0)
        y1 = y0 + self.dt * self.v * np.sin(theta0)
        z1 = 0
        theta1 = theta0 + u

        return x1, y1, z1, theta1, False
    
    def curvePrimitives_justToShow(self, x0, y0, theta0, u):
        #It returns the sequence of steps within a single input
        t = np.arange(0, self.t_max, self.dt)
        lengthT = len(t)
        x = [x0]
        y = [y0]
        z = [0]
        theta = [theta0]

        for _ in range(lengthT):
            x_next, y_next, z_next, theta_next, isInObstacle = self.curvePrimitives_singleStep_justToShow(x[-1], y[-1], theta[-1], u)
            if not isInObstacle:
                x.append(x_next)
                y.append(y_next)
                z.append(z_next)
                theta.append(theta_next)
    
        return x,y,z,theta


if __name__ == "__main__":
    simulator = DubinSimulator()
    ## Initialization of variables
    #initial state
    x0 = 0
    y0 = 0
    theta0 = 0
    #the list containing the single states
    list = []
    #the input
    max_input = 0.25
    step_input = 0.01

    ## Here I change the input
    for ii in np.arange(0,max_input,step_input): 
        innerlist = []
        #left input
        x,y,z,theta = simulator.curvePrimitives(x0, y0, theta0, ii) #Replace with simulator.curvePrimitives_justToShow for plotting
        innerlist = [x,y,z]
        list.append(innerlist)
        #right input
        x,y,z,theta = simulator.curvePrimitives(x0, y0, theta0, -ii)    #Replace with simulator.curvePrimitives_justToShow for plotting
        innerlist = [x,y,z]
        list.append(innerlist)
    
    ## Here I draw the lines in a 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for x_val, y_val, z_val in list:
        ax.plot(x_val, y_val, z_val)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()