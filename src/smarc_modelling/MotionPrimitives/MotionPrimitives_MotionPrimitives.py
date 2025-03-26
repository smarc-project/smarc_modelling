import numpy as np
import matplotlib.pyplot as plt
import math
import smarc_modelling.MotionPrimitives.GlobalVariables_MotionPrimitives as glbv
import sys
sys.path.append('~/Desktop/smarc_modelling-master')
from smarc_modelling.vehicles.SAM import SAM
from mpl_toolkits.mplot3d import Axes3D
from smarc_modelling.MotionPrimitives.ObstacleChecker_MotionPrimitives import *
import math


class SAM_PRIMITIVES():
    def __init__(self):

        # 1 # select the duration of 1 step within a primitive 
        self.dt = glbv.DT_PRIMITIVES
        
        # 2 # select the duration of 1 primitive 
        self.t_span = (0, 3)

        # 3 # do not touch!
        self.n_sim = int(self.t_span[1]/self.dt)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], self.n_sim)

        # Create SAM instance
        self.sam = SAM(self.dt)

    def dynamics_wrapper(self, x, ds_inputs, indexes):
        """
        u: control inputs as [x_vbs, x_lcg, delta_s (rad), delta_r (rad), rpm1, rpm2]
        index: 2 for vertical primitives and 3 for horizontal primitives
        """

        # Default conditions
        u = np.zeros(6)
        u[0] = 50  # VBS
        u[1] = 50   # LCG
        #u[2] = np.deg2rad(ds_input)   # Vertical (stern)--aileron // IN RADIANTS //ds_input
        #u[3] = np.deg2rad(ds_input)    # Horizontal (rudder) // IN RADIANTS
        #u[4] = 500     # RPM 1
        u[5] = u[4]     # RPM 2

        # Unpacking the inputs the user wants, transforming angles in radiants if necessary
        for ii in range(len(indexes)):
            if int(indexes[ii]) == 2 or int(indexes[ii]) == 3:
                u[int(indexes[ii])] = np.deg2rad(ds_inputs[ii])
            elif int(indexes[ii]) == 4: 
                u[int(indexes[ii])] = ds_inputs[ii]
                u[int(indexes[ii]+1)] = ds_inputs[ii]
            else:
                u[int(indexes[ii])] = ds_inputs[ii]

        return self.sam.dynamics(x, u)

    def curvePrimitives_singleStep(self, x, ds_inputs, indexes):
        '''
        dynamical model with forward Euler, it returns a SINGLE step within one primitive and the cost for such step
        '''

        data = np.empty(len(x)) 
        data[:] = x + self.dynamics_wrapper(x, ds_inputs, indexes) * self.dt 
        cost = self.computeCost(x, data[:])

        return data, cost
    
    def computeCost(self, x0, x1):
        """
        This function computes the cost of each single step within one primitive (It will be the g_cost)
        """

        # 1-Using the distance from node(k-1) and node(k)
        cost = np.sqrt((x0[0] - x1[0])**2 + (x0[1] - x1[1])**2 + (x0[2] - x1[2])**2)
        
        # 2-Using the current acceleration 
        #v0 = np.sqrt(x0[7]**2 + x0[8]**2 + x0[9]**2)
        #v1 = np.sqrt(x1[7]**2 + x1[8]**2 + x1[9]**2)
        #cost = ((v1 + v0) * self.dt) * 0.5

        return cost
        
    def curvePrimitives(self, x0, ds_inputs, indexes_u, map_instance, angle, breaking = False):
        '''
        It returns the sequence of steps within a single input primitive.

        The output will be (a, b, c, d), where:
            a: sequence of point within one primitive (one single input)    --> We use them to plot the primitive (we can not only use the last point, otherwise it will be a stright line)
            b: the cost of this path, from x0 to x1 (within one single input)   --> We add it to the cost of the previous node (x0)
            c: True if at least one point of the primitive lies in an obstacle, False otherwise
            d: True if at least one point of the primitive lies in the goal area, False otherwise
        '''

        # Compute the dynamical value for the primitiveLength
        maxValue = 3 
        minValue = 1.5
        stepAngle = 85
        MinAngle = np.min([angle, np.pi- angle])
        if np.rad2deg(MinAngle) < stepAngle:
            # Increasing exponential
            #computedSpan = minValue*np.exp(np.log(maxValue/minValue)*np.rad2deg(MinAngle)*1/stepAngle)

            # Decreasing exponential
            computedSpan = maxValue*np.exp(np.log(minValue/maxValue)*np.rad2deg(MinAngle)*1/stepAngle)
        else:
            # Incr
            #computedSpan = maxValue

            # Decr
            computedSpan = minValue

        # Compute the dynamic t_span
        self.t_span = (0, computedSpan)
        self.n_sim = int(self.t_span[1]/self.dt)

        # Initialize the variables
        cost_sum = 0
        data = np.empty((len(x0), self.n_sim))  # a matrix containing for each state in x0, n_sim values (empty rn)
        data[:, 0] = x0
        arrivedPointBefore = False
        finalState = None 

        # Compute the states within one primitive and checking if they lie in obstacles or goal area
        for i in range(self.n_sim - 1):
            if not arrivedPointBefore:
                data[:, i+1], cost = self.curvePrimitives_singleStep(data[:, i], ds_inputs, indexes_u)
            else:
                data[:, i+1] = finalState
            cost_sum += cost

            if breaking:
                # Check velocity
                q0,q1,q2,q3,vx,vy,vz = data[3:10, i+1]
                current_v = body_to_global_velocity((q0,q1,q2,q3), [vx,vy,vz])
                current_v_norm = np.linalg.norm(current_v)
                if current_v_norm <= 0.05:
                    angle = calculate_angle_goalVector(data[:, i+1], current_v, map_instance)
                    ds_inputs = [50,50,0,0,0,0]
                    indexes_u = [0,1,2,3,4,5]

            # Find point base, A and B
            pointA = compute_A_point_forward(data[:, i+1])
            pointB = compute_B_point_backward(data[:, i+1])
            current_cg = (data[0,i+1], data[1,i+1], data[2,i+1])

            # If outside the map, reject the primitive
            if  IsOutsideTheMap(pointB[0], pointB[1], pointB[2], map_instance) or IsOutsideTheMap(pointA[0], pointA[1], pointA[2], map_instance):
                return [], -1, True, False, None

            # If arrived at the goal
            if not arrivedPointBefore and (arrived(current_cg, map_instance) or arrived(pointA, map_instance) or arrived(pointB, map_instance)):
                arrivedPointBefore = True
                finalState = data[:, i+1]
                
        return data, cost_sum, False, arrivedPointBefore, finalState

    def curvePrimitives_justToShow(self, x0, ds_inputs, indexes_u):
        '''
        ONLY USED FOR PLOTTING THE PRIMITIVES IN THIS SCRIPT

        It returns the sequence of steps within a single input.
        The output will be (a, b, c, d), where:
            a: sequence of point within one primitive (one single input)    --> We use them to plot the primitive (we can not only use the last point, otherwise it will be a stright line)
            b: the cost of this path, from x0 to x1 (within one single input)   --> We add it to the cost of the previous node (x0)
            c: True if at least one point of the primitive lies in an obstacle, False otherwise
            d: True if at least one point of the primitive lies in the goal area, False otherwise
        '''

        # Initialize the variables
        cost = 0
        data = np.empty((len(x0), self.n_sim))  #A matrix containing for each state in x0, n_sim values (empty rn)
        data[:, 0] = x0
        
        # Computing the single steps for one single input primitive
        for i in range(self.n_sim - 1):
            data[:, i+1], cost = self.curvePrimitives_singleStep(data[:, i], ds_inputs, indexes_u)

        return data, cost   

if __name__ == "__main__":
    '''
    This main script serves for plotting only the primitives we are using. You can modify the primitives here only to show them!
    ########
    # MIND # that changing the type of primitives here will not change the type of primitives in the main algorithm
    ########
    '''

    # Initialize the class
    simulator = SAM_PRIMITIVES()

    # Initial conditions
    eta0 = np.zeros(7)
    eta0[3] = 1.0       # Initial quaternion (no rotation) 
    nu0 = np.zeros(6)   # Zero initial velocities
    u0 = np.zeros(6)    # The initial control inputs for SAM
    u0[0] = 50  #Vbs
    u0[1] = 50  #lcg
    x0 = np.concatenate([eta0, nu0, u0])

    # Initial conditions for the input
    max_input = 7
    step_input = 2

    ### Defining the inputs with meshgrid (faster than two for loops) --> inputs = (values, indices) --> full_input_pairs = np.array([500, 4]) for instance: 500RPM to u[4]
    #1# HOW MANY INPUTS ARE YOU USING? ###
    nInputs = 5

    #2# WHAT ARE THE INPUTS THAT YOU ARE USING? ###
    rudder_inputs = np.arange(-max_input, max_input, step_input)
    stern_inputs = np.array([-7, 0, 7])
    vbs_inputs = np.array([10, 50, 90])
    lcg_inputs = np.array([0, 50, 100])
    rpm_inputs = np.arange(-1000, 1000, 200)

    #3# ADD THE NAME OF YOUR INPUT INSIDE np.meshgrid() ###
    input_pairs = np.array(np.meshgrid(rudder_inputs, rpm_inputs, stern_inputs, vbs_inputs, lcg_inputs)).T.reshape(-1,nInputs)

    #4# ADD THE INDEX IN U RELATED TO YOUR INPUT (ex. modifying rpm --> np.tile([4], leave it)###
    additional_values = np.tile([3, 4, 2, 0, 1], (input_pairs.shape[0], 1))

    # Generate the primitives (not parallelised)
    full_input_pairs = np.hstack((input_pairs, additional_values))

    list = []
    print("Computing...")
    for inputs in full_input_pairs:
        innerlist = []
        inputLen = len(inputs)
        data, cost = simulator.curvePrimitives_justToShow(x0, inputs[0 : inputLen//2], inputs[inputLen//2 : inputLen])
        innerlist = [data[0,:],data[1,:],data[2,:]]
        list.append(innerlist)

    # Draw the lines in the 3D space
    print("Drawing...")
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for x_val, y_val, z_val in list:
        ax.plot(x_val, y_val, z_val)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set aspect ratio to be equal
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
    
    # Plot 3D surface
    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()
    x_grid = np.arange(x_min-20, x_max+20)
    y_grid = np.arange(y_min-20, y_max+20)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0*xx
    ax.plot_surface(xx, yy, zz, alpha=0.3)

    # Set axes equal
    set_axes_equal(ax)

    # Show
    print("DONE!")
    plt.show()
    