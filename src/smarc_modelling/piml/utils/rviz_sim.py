#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from piml_msgs.msg import SynchedData
import numpy as np
from smarc_modelling.vehicles.SAM_PIML import SAM_PIML
from nav_msgs.msg import Odometry

class RvizSim(Node):

    def __init__(self):
        super().__init__("rviz_sim")

        self.need_init_pose = True

        # Subscriber to the synched topic
        self.sync_sub = self.create_subscription(SynchedData, "/synched_data", self.call_sim, 1)

        # Publisher for new odom message
        self.odom_pub = self.create_publisher(Odometry, "/piml/rviz/odom", 1)
    
    def call_sim(self, msg):

        # We have init pose so we want to do a time step with the sim
        if self.need_init_pose == False:
            
            # Creating the sim message for rviz
            sim_msg = Odometry()
            sim_msg.header.frame_id = "mocap"
            sim_msg.child_frame_id = "sam_mocap2/base_link"
            sim_msg.header.stamp.sec = self.sec
            sim_msg.header.stamp.nanosec = self.nanosec

            # Get dt from new msg difference in time
            self.sec = msg.lcg_cmd.header.stamp.sec
            self.nanosec = msg.lcg_cmd.header.stamp.nanosec
            new_time = self.sec + self.nanosec * 1e-9
            dt = new_time - self.t
            dt = 0.01
            self.t = new_time


            # Simulator time step using rk4
            self.pose = self.rk4(self.pose, self.controls, dt, self.sam.dynamics)

            # Normalize quaternion to help with stability when sim is misbehaving
            quat = self.pose[2:6]
            quat = quat / np.linalg.norm(quat)
            self.pose[3] = quat[0]
            self.pose[4] = quat[1]
            self.pose[5] = quat[2]
            self.pose[6] = quat[3]

            # Publish sim results for use in Rviz
            nu = self.pose[0:7]
            sim_msg.pose.pose.position.x = nu[0]
            sim_msg.pose.pose.position.y = nu[1]
            sim_msg.pose.pose.position.z = nu[2]
            sim_msg.pose.pose.orientation.x = nu[3]
            sim_msg.pose.pose.orientation.y = nu[4]
            sim_msg.pose.pose.orientation.z = nu[5]
            sim_msg.pose.pose.orientation.w = nu[6]
    
            eta = self.pose[7:13]
            sim_msg.twist.twist.linear.x = eta[0]
            sim_msg.twist.twist.linear.y = eta[1]
            sim_msg.twist.twist.linear.z = eta[2]
            sim_msg.twist.twist.angular.x = eta[3]
            sim_msg.twist.twist.angular.y = eta[4]
            sim_msg.twist.twist.angular.z = eta[5]

            self.odom_pub.publish(sim_msg)

            # Get new control values for the next time step
            vbs_cmd = msg.vbs_cmd.value
            lcg_cmd = msg.lcg_cmd.value
            dS = msg.thrust_vector_cmd.thruster_vertical_radians
            dR = msg.thrust_vector_cmd.thruster_horizontal_radians 
            rpm1 = msg.thruster1_cmd.rpm 
            rpm2 = msg.thruster2_cmd.rpm 
            self.controls = [vbs_cmd, lcg_cmd, dS, dR, rpm1, rpm2]

        # First time we get data we set the init pose and start the sim
        elif self.need_init_pose == True:
            
            # Pull out init data from msg

            # Time
            self.sec = msg.lcg_cmd.header.stamp.sec
            self.nanosec = msg.lcg_cmd.header.stamp.nanosec
            self.t = self.sec + self.nanosec * 1e-9

            # Pose
            x = msg.odom_gt.pose.pose.position.x
            y = msg.odom_gt.pose.pose.position.y
            z = msg.odom_gt.pose.pose.position.z
            q1 = msg.odom_gt.pose.pose.orientation.x
            q2 = msg.odom_gt.pose.pose.orientation.y
            q3 = msg.odom_gt.pose.pose.orientation.z
            q4 = msg.odom_gt.pose.pose.orientation.w
            eta = [x, y, z, q1, q2, q3, q4]

            # Speeds
            u = msg.odom_gt.twist.twist.linear.x
            v = msg.odom_gt.twist.twist.linear.y
            w = msg.odom_gt.twist.twist.linear.z
            p = msg.odom_gt.twist.twist.angular.x
            q = msg.odom_gt.twist.twist.angular.y
            r = msg.odom_gt.twist.twist.angular.z
            nu = [u, v, w, p, q, r]

            # Actuator positions
            vbs_fb = msg.vbs_fb.value
            lcg_fb = msg.lcg_fb.value
            dS = msg.thrust_vector_cmd.thruster_vertical_radians # Has no fb published so we take cmd
            dR = msg.thrust_vector_cmd.thruster_horizontal_radians # -||-
            rpm1 = msg.thruster1_cmd.rpm # At the moment fb for rpms are not correct so we take the cmd
            rpm2 = msg.thruster2_cmd.rpm # -||-
            u = [vbs_fb, lcg_fb, dS, dR, rpm1, rpm2]

            self.pose = np.concatenate([eta, nu, u])

            # Getting the first actuator commands
            vbs_cmd = msg.vbs_cmd.value
            lcg_cmd = msg.lcg_cmd.value
            # dS, dR, rpm1 & rpm2 are already the cmd values since we don't have fb for them
            self.controls = [vbs_cmd, lcg_cmd, dS, dR, rpm1, rpm2]

            # SAM instance for predicted dynamics NOTE: Dynamics update of dt in sam? Maybe add a change dt function
            self.sam = SAM_PIML(dt=0.02, piml_type=None)

            # Update init pose status
            self.need_init_pose = False

    def rk4(self, x, u, dt, fun):
        # https://github.com/smarc-project/smarc_modelling/blob/master/src/smarc_modelling/sam_sim.py#L38C1-L46C15 
        k1, _, _ = fun(x, u) # (19, )
        k2, _, _ = fun(x+dt/2*k1, u)
        k3, _, _ = fun(x+dt/2*k2, u)
        k4, _, _ = fun(x+dt*k3, u)
        x_t = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x_t
    
    def ef(self, x, u, dt, fun):
        # Euler forward
        dx, _, _ = fun(x, u)
        x_t = x + dx * dt
        return x_t

def main(args=None):
    # Start and run node
    rclpy.init(args=args)

    # Node setup
    node_rvizsim = RvizSim()
    rclpy.spin(node_rvizsim)

    # Safe exit
    node_rvizsim.destroy_node()



if __name__ == "__main__":
    main()