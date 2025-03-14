import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('~/Desktop/smarc_modelling-master')
from smarc_modelling.vehicles.SAM import SAM
from mpl_toolkits.mplot3d import Axes3D
from smarc_modelling.MotionPrimitivesTest.ObstacleChecker import IsWithinObstacle

'''
This script defines the Motion primitives for SAM.
It is called by the MainScript.py to draw the primitives. If you want to draw the primitives locally, just 
edit this script changing curvePrimitives() with curvePrimitives_justToShow()

The __main__ plots the local primitives
'''

class SAM_PRIMITIVES():
    def __init__(self):
        # Initial conditions
        self.eta0 = np.zeros(7)
        self.eta0[3] = 1.0  # Initial quaternion (no rotation) 
        self.nu0 = np.zeros(6)  # Zero initial velocities
        self.u0 = np.zeros(6)    #The initial control inputs for SAM
        self.u0[0] = 50  #Vbs
        self.u0[1] = 50  #lcg
        self.x0 = np.concatenate([self.eta0, self.nu0, self.u0])

        # Simulation timespan
        self.dt = 0.1
        self.t_span = (0, 20)
        self.n_sim = int(self.t_span[1]/self.dt)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], self.n_sim)

        # Create SAM instance
        self.sam = SAM(self.dt)
    
    def dynamics_wrapper(self, t, x, ds_input):
        """
        u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        """
        # These are used (modify here if needed)
        u = np.zeros(6)
        u[0] = 50  # VBS
        u[1] = 50   # LCG
        #u[2] = np.deg2rad(ds_input)   # Vertical (stern)--aileron // IN RADIANTS //ds_input
        u[3] = np.deg2rad(ds_input)    # Horizontal (rudder) // IN RADIANTS
        u[4] = 1500     # RPM 1
        u[5] = u[4]     # RPM 2

        return self.sam.dynamics(x, u)

    def curvePrimitives_singleStep(self, x, ds_input):
        '''
        dynamical model with forward Euler, it returns a SINGLE step
        '''

        data = np.empty(len(x))  #A matrix containing for each state in x0, n_sim values (empty rn)
        data[:] = x + self.dynamics_wrapper(self.dt, x, ds_input) * (self.t_span[1]/self.n_sim) 

        return data
    
    def curvePrimitives(self, x0, ds_input):
        '''
        It returns the sequence of steps within a single input
        '''
        cost = 0
        data = np.empty((len(x0), self.n_sim))  #A matrix containing for each state in x0, n_sim values (empty rn)
        data[:, 0] = x0
        for i in range(self.n_sim - 1):
            data[:, i+1] = self.curvePrimitives_singleStep(data[:, i], ds_input)
            #check if it is within an obstacle
            if IsWithinObstacle(data[0,i+1], data[1,i+1]):
                return [], -1
            #the cost of this new path = squared distance from node(k-1) and node(k)
            cost = cost + (data[0,i] - data[0, i+1])**2 + (data[1,i] - data[1, i+1])**2
        
        return data, cost  #containing all the states within a single input (one single primitive)
    
    def curvePrimitives_singleStep_justToShow(self, x, ds_input):
        #dynamical model with forward Euler, it returns a SINGLE step
        data = np.empty(len(self.x0))  #A matrix containing for each state in x0, n_sim values (empty rn)
        data[:] = x + self.dynamics_wrapper(self.dt, x, ds_input) * (self.t_span[1]/self.n_sim) 

        return data
    
    def curvePrimitives_justToShow(self, ds_input):
        #It returns the sequence of steps within a single input
        #It returns the sequence of steps within a single input
        data = np.empty((len(self.x0), self.n_sim))  #A matrix containing for each state in x0, n_sim values (empty rn)
        data[:,0] = self.x0
        for i in range(self.n_sim - 1):
            data[:,i+1] = self.curvePrimitives_singleStep_justToShow(data[:,i], ds_input)
            #check if z is negative!
            if data[2,i+1]>0:
                data[2,i+1] = 0

        return data  


if __name__ == "__main__":
    simulator = SAM_PRIMITIVES()

    #the list containing the single states
    list = []

    #the input
    max_input = 6
    step_input = 1
    
    ## Here I change the input
    #for ii in np.arange(-max_input,max_input, step_input): 
    for ii in np.arange(-max_input,max_input + step_input,step_input): 
        innerlist = []
        data = simulator.curvePrimitives_justToShow(ii) #Replace with simulator.curvePrimitives_justToShow for plotting
        innerlist = [data[0,:],data[1,:],data[2,:]]
        list.append(innerlist)

    ## Here I draw the lines in a 3D space
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for x_val, y_val, z_val in list:
        ax.plot(x_val, y_val, z_val)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ## Set aspect ratio to be equal
    def set_axes_equal(ax):
        """Set equal scaling for 3D plots, ensuring 1:1:1 aspect ratio."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        max_range = np.array([x_range, y_range, z_range]).max() / 2.0

        mid_x = (x_limits[0] + x_limits[1]) / 2.0
        mid_y = (y_limits[0] + y_limits[1]) / 2.0
        mid_z = (z_limits[0] + z_limits[1]) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ## Plot 2D surface for z = 0
    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min-20, x_max+20)
    y_grid = np.arange(y_min-20, y_max+20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx
    ax.plot_surface(xx, yy, zz, alpha=0.3)

    set_axes_equal(ax)
    plt.show()
    