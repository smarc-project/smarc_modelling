#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# TODO: Check if the stern/rudder direction are good compare with bag in rviz

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from message_filters import Subscriber, ApproximateTimeSynchronizer

from smarc_msgs.msg import PercentStamped, ThrusterRPM, ThrusterFeedback
from sam_msgs.msg import ThrusterAngles
from piml_msgs.msg import SynchedData, ThrusterRPMStamped
from nav_msgs.msg import Odometry

# Some of the messages are missing timestamp so we make new ones
class AddTimestamp(Node):
    def __init__(self):
        super().__init__('add_timestamps')

        # NOTE: Since we reassign the stamps for some of the topics we need to do it to all topics we are going to use
        # so that they exist in the same "timeline"

        # Creating publishers
        # NOTE: Get the thrust vectoring
        self.thruster1_cmd_pub = self.create_publisher(ThrusterRPMStamped, "/piml/thruster1_cmd", 10)
        self.thruster2_cmd_pub = self.create_publisher(ThrusterRPMStamped, "/piml/thruster2_cmd", 10)
        self.thruster1_fb_pub = self.create_publisher(ThrusterFeedback, "/piml/thruster1_fb", 10) 
        self.thruster2_fb_pub = self.create_publisher(ThrusterFeedback, "/piml/thruster2_fb", 10)
        self.lcg_cmd_pub = self.create_publisher(PercentStamped, "/piml/lcg_cmd", 10)
        self.lcg_fb_pub = self.create_publisher(PercentStamped, "/piml/lcg_fb", 10)
        self.vbs_cmd_pub = self.create_publisher(PercentStamped, "/piml/vbs_cmd", 10)
        self.vbs_fb_pub = self.create_publisher(PercentStamped, "/piml/vbs_fb", 10)
        self.odom_pub = self.create_publisher(Odometry, "/piml/odom", 10)
        self.thrust_vectoring_pub = self.create_publisher(ThrusterAngles, "/piml/thrust_vector_cmd", 10)
        
        # Subscribe to topics we want to add timestamps to
        self.thruster1_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster1_cmd", self.add_stamp_thruster1, 1) # No stamp at all
        self.thruster2_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster2_cmd", self.add_stamp_thruster2, 1) # No stamp at all
        self.thruster1_fb_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster1_fb", self.add_stamp_thruster1fb, 1) # No data in stamp 
        self.thruster2_fb_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster2_fb", self.add_stamp_thruster2fb, 1) # No data in stamp
        self.lcg_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/lcg_cmd", self.add_stamp_lcg_cmd, 1) # No data in stamp
        self.lcg_fb_sub = self.create_subscription(PercentStamped, "/sam/core/lcg_fb", self.add_stamp_lcg_fb, 1) # Has data in stamp but need to reassign
        self.vbs_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/vbs_cmd", self.add_stamp_vbs_cmd, 1) # No data in stamp
        self.vbs_fb_sub = self.create_subscription(PercentStamped, "/sam/core/vbs_fb", self.add_stamp_vbs_fb, 1) # Has data in stamp but need to reassign
        # /mocap/sam_mocap/odom is sometimes called /mocap/sam_mocap/odom if you are not getting any sync messages
        self.odom_sub = self.create_subscription(Odometry, "/mocap/sam_mocap2/odom", self.add_stamp_odom, 1) # Has data in stamp but need to reassign
        self.thrust_vectoring_sub = self.create_subscription(ThrusterAngles, "/sam/core/thrust_vector_cmd", self.add_stamp_vector, 1) # Has dada in stamp but need to reassign

    # A bit heavily hard coded but whatevs 
    def add_stamp_odom(self, msg):
        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.odom_pub.publish(msg_stamped)

    def add_stamp_vbs_fb(self, msg):
        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.vbs_fb_pub.publish(msg_stamped)

    def add_stamp_vbs_cmd(self, msg):
        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.vbs_cmd_pub.publish(msg_stamped)

    def add_stamp_lcg_fb(self, msg):
        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.lcg_fb_pub.publish(msg_stamped)

    def add_stamp_lcg_cmd(self, msg):
        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.lcg_cmd_pub.publish(msg_stamped)

    def add_stamp_thruster2fb(self, msg):
        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.thruster2_fb_pub.publish(msg_stamped)

    def add_stamp_thruster1fb(self, msg):

        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()

        self.thruster1_fb_pub.publish(msg_stamped)

    def add_stamp_thruster1(self, msg):
        msg_stamped = ThrusterRPMStamped()
        msg_stamped.rpm = msg.rpm
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.thruster1_cmd_pub.publish(msg_stamped)
    
    def add_stamp_thruster2(self, msg):
        msg_stamped = ThrusterRPMStamped()
        msg_stamped.rpm = msg.rpm
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.thruster2_cmd_pub.publish(msg_stamped)

    def add_stamp_vector(self, msg):
        msg_stamped = ThrusterAngles()
        msg_stamped.thruster_vertical_radians = msg.thruster_vertical_radians
        msg_stamped.thruster_horizontal_radians = msg.thruster_horizontal_radians
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.thrust_vectoring_pub.publish(msg_stamped)


def main(args=None):
    # Start and run node
    rclpy.init(args=args)
    
    # Node setup
    node_stamp = AddTimestamp()

    exec = MultiThreadedExecutor()
    
    try:
        exec.add_node(node_stamp)
        exec.spin()
    except:
        # Safe exit
        node_stamp.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()