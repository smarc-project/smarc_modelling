
import heapq
import numpy as np
import sys
import random
from joblib import Parallel, delayed
from threading import Lock
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import time
import multiprocessing
import csv
# from sklearn import tree
from sklearn import tree
from smarc_modelling.motion_planning.MotionPrimitives.MotionPrimitives import SAM_PRIMITIVES
from smarc_modelling.motion_planning.MotionPrimitives.ObstacleChecker import compute_B_point_backward, calculate_angle_goalVector, compute_A_point_forward
from smarc_modelling.motion_planning.MotionPrimitives.Optimizer.acados_trajectory_simulator import join_trees_optimization
from smarc_modelling.motion_planning.MotionPrimitives.trm_colors import *
import smarc_modelling.motion_planning.MotionPrimitives.GlobalVariables as glbv
import matplotlib.pyplot as plt
from smarc_modelling.motion_planning.MotionPrimitives.ObstacleChecker import *

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi

## Class for the path planner. Not a ros node itself, but it will be used by the ros node. It contains the main logic of the path planner.
## Inputs are the current state and the goal, and the output is a trajectory to follow.
class SAMPlanner():
    def __init__(self):
        self.start_state = None
        self.goal_state = None
        self.planning = False
        self.planner_mutex = Lock()
        self.map_instance = None
        # self.sim = SAM_PRIMITIVES()

        # Planner parameters
        self.threshold_yaw_diff = np.deg2rad(7) # If the yaw difference is larger than this threshold, we perform a turbo turn in place
        self.rudder_max_angle = np.deg2rad(7)
        self.rpm_fwd = 500
        self.rpm_bwd = -600
        self.n_sim = 30 # length of primitive

        # MPC setup
        self.dt = 0.1
        self.sam = SAM_casadi(dt = self.dt)
        self.dynamics = self.sam.dynamics() 

        # For rviz visualization
        self.tree = []
        self.trajectory = []
        self.branch_cnt = 0

    def instantiate(self, current, goal, map_instance):
        self.start_state = current
        self.goal_state = goal
        self.map_instance = map_instance

    def plan_path(self):
        # Correct heading from start to goal if needed
        tree_number = 1
        primitive_tree1, status1 = self.correct_heading(self.start_state, self.goal_state, self.map_instance, tree_number)
        print("Status of heading correction from start to goal:", status1)
        
        # Correct heading from goal to start if needed
        tree_number = 2
        primitive_tree2, status2 = self.correct_heading(self.goal_state, self.start_state, self.map_instance, tree_number)
        print("Status of heading correction from goal to start:", status2)

        # If one of the trees did not need to execute a turbo turn, the primitive_tree has only one state

        # Connect the two trees if they were generated successfully
        final_path = []
        status = "planning_failed"
        connections = []
        if primitive_tree1 is not None and primitive_tree2 is not None:
            # Select best nodes for connecting the trees
            connection_1 = primitive_tree1[0:3,-1] if status1 == "turbo_turn" else primitive_tree1[0:3]
            connection_2 = primitive_tree2[0:3,-1] if status2 == "turbo_turn" else primitive_tree2[0:3]
            connections.append(connection_1.copy())
            connections.append(connection_2.copy())

            connecting_path, status = self.connect_trees(primitive_tree1, primitive_tree2)
            # status = False

            # Nacho: put together final path here
            final_path = connecting_path

        return final_path, status, self.tree, connections

    def connect_trees(self, primitive_tree1, primitive_tree2):        

        # Prepare the waypoints on the second tree for optimization
        print(primitive_tree1[:,-1])
        print(primitive_tree2[:,-1])
        waypoints = []

        # Option 1: connect and optimize second path
        # for i in range(0, primitive_tree2.shape[1]):
        #     reverted_waypoint = primitive_tree2[:,i].copy()
        #     reverted_waypoint[7:13] = -reverted_waypoint[7:13]  # Invert velocities
        #     reverted_waypoint[17:] = -reverted_waypoint[17:]   # Invert rpm 
        #     waypoints.append(reverted_waypoint)               

        # for _ in range(50):
        #     waypoints.append(primitive_tree2[:,-1].copy())
        # waypoints.append(primitive_tree1[:,-1].copy())
        # array_waypoints = np.asarray(waypoints[::-1])

        # Option 2: just connect the trees
        waypoints.append(primitive_tree1[:,-1].copy())
        for _ in range(50):
            reverted_waypoint = primitive_tree2[:,-1].copy()
            reverted_waypoint[7:13] = -reverted_waypoint[7:13]  # Invert velocities
            reverted_waypoint[17:] = -reverted_waypoint[17:]   # Invert rpm 
            waypoints.append(reverted_waypoint)

        array_waypoints = np.asarray(waypoints)

        # Connect the two trees and optimize the inverted second tree waypoints
        N_hor = 30
        T_s = 0.1
        build = True
        optimized_waypoints, status = join_trees_optimization(array_waypoints, N_hor, T_s, build)

        if status == 0:
            return optimized_waypoints, "trees_connected"
        else:   
            print("Optimization failed with status:", status)
            return [], "optimization_failed"

    def correct_heading(self, current, goal, map_instance, tree_number):

        ## From the current state to the goal
        # Check if the goal can be reached given its distance and the maximum yaw change that can be achieved in one primitive. 
        # If not, perform a turbo turn in place to reduce the yaw difference and get closer to the goal.
        yaw_diff = self.compute_heading_diff(current, goal, tree_number)
        print(f"Initial yaw difference: {np.rad2deg(yaw_diff):.2f} degrees in tree {tree_number}")
        if np.abs(yaw_diff) > self.threshold_yaw_diff: 
            direction = "right" if np.sign(yaw_diff) > 0 else "left"
            primitive_tree, rejected = self.turbo_turn_in_place(current, goal, yaw_diff, map_instance, 
                                                                direction, tree_number)
            # Collision detected during turbo turn
            if rejected:
                return None, "collision"
            else:
                return primitive_tree, "turbo_turn"
        else:
            # No turbo turn needed
            return current, "no_turbo_turn"

    def rk4(self, x, u, dt, fun):
        k1 = fun(x, u)
        k2 = fun(x+dt/2*k1, u)
        k3 = fun(x+dt/2*k2, u)
        k4 = fun(x+dt*k3, u)
        x_t = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
      
        return x_t.full().flatten()

    def turbo_turn_in_place(self, current, goal, yaw_diff, map_instance, direction, tree_number):

        data = np.empty((len(current), 1))
        data[:, 0] = current

        # Generic turbo turn ctrl
        # "vbs", "lcg", "ds", "dr", "rpm_1", "rpm_2"
        u = np.array([50, 50, 0, self.rudder_max_angle, self.rpm_fwd, self.rpm_fwd])

        # Set the rudder angle and the thruster commands based on the direction of the turn        
        u[3] = self.rudder_max_angle if direction == "left" else -self.rudder_max_angle
        # Start with forward motion 
        u[4] = u[5] = self.rpm_fwd

        # Compute the states within one primitive and checking if they lie in obstacles or goal area
        i = 0
        print(f"Performing turbo turn in place to the {direction} in tree {tree_number}")
        while np.abs(yaw_diff) > self.threshold_yaw_diff:
            # Normalize quaternion in current state to avoid numerical issues in the dynamics
            q = data[3:7, -1]
            q = q / (np.linalg.norm(q) + 1e-12)
            data[3:7, -1] = q
            
            # Simulate the primitive
            primitive_idx = i * self.n_sim
            # print(f"Primitive {primitive_idx} with u {u}")
            for j in range(0, self.n_sim):
                data = np.concatenate((data, np.empty((len(current), 1))), axis=1)
                data[:,primitive_idx + j + 1] = self.rk4(data[:, primitive_idx + j], u, self.dt, self.dynamics)
                self.tree = self.add_branch(self.tree, tree_number, self.branch_cnt, 
                            data[:, primitive_idx + j], 
                            data[:, primitive_idx + j + 1])
                self.branch_cnt += 1
                
            # Collision detection
            pointA = compute_A_point_forward(data[:, primitive_idx + self.n_sim - 1])
            pointB = compute_B_point_backward(data[:, primitive_idx + self.n_sim - 1])
            # If outside the map, reject the primitive
            if  IsOutsideTheMap(pointB[0], pointB[1], pointB[2], map_instance) or \
                IsOutsideTheMap(pointA[0], pointA[1], pointA[2], map_instance): 
                    if i == 0:
                        # If the point forward in obstacle, try backwards motion
                        u[3] *= -1 
                        u[4] *= -1 
                        u[5] = u[4] 
                        i += 1
                        continue
                    else:
                        # If both points forward and backwards in obstacle, the vehicle is trapped
                        # and we need a different type of primitive to get out of the obstacle. 
                        return [], True
            
            # Alternate the rudder angle to perform a turn in place
            u[3] *= -1 
            # Alternate the direction of the thrusters to perform a turn in place
            u[5] = u[4] = self.rpm_bwd if np.sign(u[4]) > 0 else self.rpm_fwd

            # Add branch to the rviz visualization tree
            # self.tree = self.add_branch(self.tree, tree_number, self.branch_cnt, 
            #                             data[:, primitive_idx], 
            #                             data[:, primitive_idx + self.n_sim - 1])
            # self.branch_cnt += 1

            # Recompute yaw difference
            yaw_diff = self.compute_heading_diff(data[:, primitive_idx + self.n_sim - 1], goal, tree_number)
            i += 1
            print(f"Yaw difference: {np.rad2deg(yaw_diff):.2f} degrees in tree {tree_number}")

        # If distance longer than a threshold, reject the primitive. Most likely the motion model failed
        # if math.hypot(data[0, -1]-data[0, 0], data[1, -1]-data[1, 0], data[2, -1]-data[2, 0]) > 1.:
        #     return [], -1, True, False, None
                
        return data, False 

    def compute_heading_diff(self, current, goal, tree_number):
        # Compute the angle between the vehicle's heading and the vector to the goal
        yaw_current = R.from_quat(np.roll(current[3:7],-1)).as_euler('xyz')[2] # [x, y, z, w] format
        # If second tree, compute angle wrt back of vehicle
        if tree_number == 2:
            yaw_current = self.angle_wrap(yaw_current + np.pi)

        vector_to_goal = goal[:3] - current[:3]
        yaw_goal_vector = np.arctan2(vector_to_goal[1], vector_to_goal[0])
        yaw_diff = self.angle_wrap(yaw_goal_vector - yaw_current)
        return yaw_diff

    def angle_wrap(self, angle):
        # Wrap angle to [-pi, pi]
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

    def add_branch(self, tree, subtree, branch_cnt, k, v):
        marker = Marker()
        def add_point(p):
            pt = Point()
            pt.x = p[0]
            pt.y = p[1]
            pt.z = p[2]
            # print(pt)
            marker.points.append(pt)
        t = time.time()
        marker.header.frame_id = "mocap"
        marker.header.stamp.sec = int(t)
        marker.header.stamp.nanosec = int((t - int(t)) * 1e9)
        marker.id = branch_cnt
        marker.type = Marker.LINE_LIST
        if subtree == 1:
            marker.color.r = 1.
            marker.color.g = 0.
            marker.color.b = 0.
        else:
            marker.color.r = 0. 
            marker.color.g = 0.
            marker.color.b = 1.
        marker.color.a = 0.5
        marker.scale.x = 0.01 
        marker.lifetime = Duration(sec=0, nanosec=0)
        add_point(k)
        add_point(v)
        tree.append(marker)
        return tree


