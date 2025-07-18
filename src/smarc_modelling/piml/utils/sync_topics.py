#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ROS imports
import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Message types
from smarc_msgs.msg import PercentStamped, ThrusterFeedback
from sam_msgs.msg import ThrusterAngles
from piml_msgs.msg import SynchedData, ThrusterRPMStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, TwistWithCovariance

class SyncSubscriber(Node):
    """Node that syncs up messages and republishes into a new packaged message"""
    def __init__(self):
        super().__init__("sync_topics")

        # Thruster doesn't actively publish so we just save their latest state
        self.thrust_vector_msg = ThrusterAngles()
        self.thrust_vector_msg.header.stamp = self.get_clock().now().to_msg()
        self.thrust_vector_msg.thruster_horizontal_radians = 0.0
        self.thrust_vector_msg.thruster_vertical_radians = 0.0

        # Tracking if we have gotten any thruster data
        self.thruster1_state = False
        self.thruster2_state = False

        # Thrusters
        self.thruster1_cmd_sub = self.create_subscription(ThrusterRPMStamped, "/piml/thruster1_cmd", self.thruster1_cmd_cb, 1) 
        self.thruster2_cmd_sub = self.create_subscription(ThrusterRPMStamped, "/piml/thruster2_cmd", self.thruster2_cmd_cb, 1) 
        self.thrust_vectoring_sub = self.create_subscription(ThrusterAngles, "/piml/thrust_vector_cmd", self.thrust_vector_cb, 1)
        self.thruster1_fb = Subscriber(self, ThrusterFeedback, "/piml/thruster1_fb") # These shouldn't be used 
        self.thruster2_fb = Subscriber(self, ThrusterFeedback, "/piml/thruster2_fb") # -||-

        # LCG & VBS
        self.lcg_cmd = Subscriber(self, PercentStamped, "/piml/lcg_cmd")
        self.lcg_fb = Subscriber(self, PercentStamped, "/piml/lcg_fb")
        self.vbs_cmd = Subscriber(self, PercentStamped, "/piml/vbs_cmd") 
        self.vbs_fb = Subscriber(self, PercentStamped, "/piml/vbs_fb")

        # Odom
        self.odom = Subscriber(self, Odometry, "/piml/odom")

        # Velocity
        self.velo = Subscriber(self, TwistStamped, "/piml/velocity")

        # The topics we want to sync
        sub_list = [self.lcg_cmd, self.lcg_fb, self.vbs_cmd, self.vbs_fb, self.thruster1_fb, self.thruster2_fb, self.odom, self.velo]

        # Set up the ApproximateTimeSynchronizer
        self.synched_message = ApproximateTimeSynchronizer(
            sub_list,
            queue_size = 50,  # How many messages we stack
            slop = 0.2 # Max time difference, seems like we can make this rather small for SAM and still get a lot of data
        )
        
        # Publisher
        self.synched_pub = self.create_publisher(SynchedData, "/synched_data", 1)
        self.synched_message.registerCallback(self.callback)

    def callback(self, lcg_cmd, lcg_fb, vbs_cmd, vbs_fb, thruster1_fb, thruster2_fb, odom, velo):

        # Making message
        sync_msg = SynchedData()

        # Check if we have been given any thruster commands
        if self.thruster1_state and self.thruster2_state:

            # Putting the correct velocity data into the odom
            twist = velo.twist
            twist_data = TwistWithCovariance()
            twist_data.twist = twist
            odom.twist = twist_data

            # Filling with collected data
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
            self.get_logger().info("Published synched data!")

    def thruster1_cmd_cb(self, msg):
        self.thruster1_cmd_msg = msg
        if not self.thruster1_state:
            self.get_logger().info("Got first set of thruster 1 controls!")
            self.thruster1_state = True

    def thruster2_cmd_cb(self, msg):
        self.thruster2_cmd_msg = msg
        if not self.thruster2_state:
            self.get_logger().info("Got first set of thruster 2 controls!")
            self.thruster2_state = True

    def thrust_vector_cb(self, msg):
        self.thrust_vector_msg = msg


def main():

    # Start and run node
    rclpy.init()

    # Node setup
    node_synch = SyncSubscriber()
    rclpy.spin(node_synch)

    # Safe exit
    node_synch.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()