#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from smarc_msgs.msg import PercentStamped, ThrusterFeedback
from sam_msgs.msg import ThrusterAngles
from piml_msgs.msg import SynchedData, ThrusterRPMStamped
from nav_msgs.msg import Odometry


# Compiles a synchronized message from SAM for easy creation of training data
class SyncSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')

        # Thrusters only sends messages when changing value so we have to exclude them from sync
        self.thrust_vector_msg = ThrusterAngles()
        self.thrust_vector_msg.header.stamp = self.get_clock().now().to_msg()
        self.thrust_vector_msg.thruster_horizontal_radians = 0.0
        self.thrust_vector_msg.thruster_vertical_radians = 0.0
        self.thruster1_state = False
        self.thruster2_state = False

        # LCG
        self.lcg_cmd = Subscriber(self, PercentStamped, "/piml/lcg_cmd")
        self.lcg_fb = Subscriber(self, PercentStamped, "/piml/lcg_fb")

        # VBS
        self.vbs_cmd = Subscriber(self, PercentStamped, "/piml/vbs_cmd") 
        self.vbs_fb = Subscriber(self, PercentStamped, "/piml/vbs_fb")

        # Thrusters
        self.thruster1_cmd_sub = self.create_subscription(ThrusterRPMStamped, "/piml/thruster1_cmd", self.thruster1_cmd_cb, 100) 
        self.thruster2_cmd_sub = self.create_subscription(ThrusterRPMStamped, "/piml/thruster2_cmd", self.thruster2_cmd_cb, 100) 
        self.thrust_vectoring_sub = self.create_subscription(ThrusterAngles, "/piml/thrust_vector_cmd", self.thrust_vector_cb, 100)
        self.thruster1_fb = Subscriber(self, ThrusterFeedback, "/piml/thruster1_fb") 
        self.thruster2_fb = Subscriber(self, ThrusterFeedback, "/piml/thruster2_fb")
    
        # Pose & Velocities
        self.odom = Subscriber(self, Odometry, "/piml/odom")

        # All the topics we want synched          
        sub_list = [self.lcg_cmd, self.lcg_fb, self.vbs_cmd, self.vbs_fb, self.thruster1_fb, self.thruster2_fb, self.odom]

        # Set up the ApproximateTimeSynchronizer
        self.synched_message = ApproximateTimeSynchronizer(
            sub_list,
            queue_size = 10,  # How many messages we stack
            slop = 0.1 # Max time difference, seems like we can make this rather small for SAM and still get a lot of data
        )

        self.synched_pub = self.create_publisher(SynchedData, "/synched_data", 10)
        self.synched_message.registerCallback(self.callback)

    def callback(self, lcg_cmd, lcg_fb, vbs_cmd, vbs_fb, thruster1_fb, thruster2_fb, odom):
        
        if self.thruster1_state and self.thruster2_state:
            # Making message
            sync_msg = SynchedData()
            sync_msg.lcg_fb = lcg_fb
            sync_msg.lcg_cmd = lcg_cmd
            sync_msg.odom_gt = odom
            sync_msg.thruster1_fb = thruster1_fb
            sync_msg.thruster2_fb = thruster2_fb
            sync_msg.thruster1_cmd = self.thruster1_cmd_msg
            sync_msg.thruster2_cmd = self.thruster2_cmd_msg
            sync_msg.thrust_vector_cmd = self.thrust_vector_msg
            sync_msg.vbs_fb = vbs_fb
            sync_msg.vbs_cmd = vbs_cmd

            # Publish message
            self.synched_pub.publish(sync_msg)
            self.get_logger().info("Published synched data")

    def thruster1_cmd_cb(self, msg):
        self.thruster1_cmd_msg = msg
        if not self.thruster1_state:
            self.get_logger().info("Got first thruster 1 controls")
            self.thruster1_state = True
    def thruster2_cmd_cb(self, msg):
        self.thruster2_cmd_msg = msg
        if not self.thruster2_state:
            self.get_logger().info("Got first thruster 2 controls")
            self.thruster2_state = True
    def thrust_vector_cb(self, msg):
        self.thrust_vector_msg = msg

    
def main(args=None):
    # Start and run node
    rclpy.init(args=args)

    # Node setup
    node_synch = SyncSubscriber()
    rclpy.spin(node_synch)

    # Safe exit
    node_synch.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()