#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from message_filters import Subscriber, ApproximateTimeSynchronizer

from smarc_msgs.msg import PercentStamped, ThrusterRPM, ThrusterFeedback
from piml_msgs.msg import SynchedData, ThrusterRPMStamped
from nav_msgs.msg import Odometry

# Some of the messages are missing timestamp so we make new ones
class AddTimestamp(Node):
    def __init__(self):
        super().__init__('add_timestamps')

        # NOTE: Since we reassign the stamps for some of the topics we need to do it to all topics we are going to use
        # so that they exist in the same "timeline"

        # Creating publishers
        self.thruster1_cmd_pub = self.create_publisher(ThrusterRPMStamped, "/piml/thruster1_cmd", 10)
        self.thruster2_cmd_pub = self.create_publisher(ThrusterRPMStamped, "/piml/thruster2_cmd", 10)
        self.thruster1_fb_pub = self.create_publisher(ThrusterFeedback, "/piml/thruster1_fb", 10) 
        self.thruster2_fb_pub = self.create_publisher(ThrusterFeedback, "/piml/thruster2_fb", 10)
        self.lcg_cmd_pub = self.create_publisher(PercentStamped, "/piml/lcg_cmd", 10)
        self.lcg_fb_pub = self.create_publisher(PercentStamped, "/piml/lcg_fb", 10)
        self.vbs_cmd_pub = self.create_publisher(PercentStamped, "/piml/vbs_cmd", 10)
        self.vbs_fb_pub = self.create_publisher(PercentStamped, "/piml/vbs_fb", 10)
        self.odom_pub = self.create_publisher(Odometry, "/piml/odom", 10)

        # Subscribe to topics we want to add timestamps to
        self.thruster1_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster1_cmd", self.add_stamp_thruster1, 10) # No stamp at all
        self.thruster2_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster2_cmd", self.add_stamp_thruster2, 10) # No stamp at all
        self.thruster1_fb_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster1_fb", self.add_stamp_thruster1fb, 10) # No data in stamp
        self.thruster2_fb_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster2_fb", self.add_stamp_thruster2fb, 10) # No data in stamp
        self.lcg_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/lcg_cmd", self.add_stamp_lcg_cmd, 10) # No data in stamp
        self.lcg_fb_sub = self.create_subscription(PercentStamped, "/sam/core/lcg_fb", self.add_stamp_lcg_fb, 10) # Has data in stamp but need to reassign
        self.vbs_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/vbs_cmd", self.add_stamp_vbs_cmd, 10) # No data in stamp
        self.vbs_fb_sub = self.create_subscription(PercentStamped, "/sam/core/vbs_fb", self.add_stamp_vbs_fb) # Has data in stamp but need to reassign
        self.odom_sub = self.create_subscription(Odometry, "/mocap/sam_mocap/odom", self.add_stamp_odom)

    # A bit heavily hard coded but whatevs 

    def add_stamp_odom(self, msg):
        msg_stamped = msg
        msg_stamped.header.stamp = self.get_clock().now().to_msg()

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


# Compiles a synchronized message from SAM for easy creation of training data
class SyncSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')

        self.thruster1_cmd_msg = ThrusterRPM
        self.thruster2_cmd_msg = ThrusterRPM
        self.lcg_cmd_msg = PercentStamped
        self.vbs_cmd_msg = PercentStamped
        self.thruster1_fb_msg = ThrusterFeedback
        self.thruster2_fb_msg = ThrusterFeedback

        # LCG
        self.lcg_cmd = Subscriber(self, PercentStamped, "/sam/core/lcg_cmd") # Missing stamp
        self.lcg_fb = Subscriber(self, PercentStamped, "/sam/core/lcg_fb")

        # VBS
        self.vbs_cmd = Subscriber(self, PercentStamped, "/sam/core/vbs_cmd") # Missing stamp
        self.vbs_fb = Subscriber(self, PercentStamped, "/sam/core/vbs_fb")

        # Thrusters
        self.thruster1_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster1_cmd", self.thruster1_cmd_cb, 10) # Missing stamp
        self.thruster2_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster2_cmd", self.thruster2_cmd_cb, 10) # Missing stamp
        self.thruster1_fb = Subscriber(self, ThrusterFeedback, "/sam/core/thruster1_fb") # Missing stamp
        self.thruster2_fb = Subscriber(self, ThrusterFeedback, "/sam/core/thruster2_fb") # Missing stamp

        self.thruster1_cmd_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster1_fb", self.thruster1_fb_cb, 10)
        self.thruster2_cmd_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster2_fb", self.thruster2_fb_cb, 10)
        self.lcg_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/lcg_cmd", self.lcg_cmd_cb, 10)
        self.vbs_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/vbs_cmd", self.vbs_cmd_cb, 10)

        # Pose & Velocities
        self.odom = Subscriber(self, Odometry, "/mocap/sam_mocap/odom")

        # All the topics we want synched
        sub_list = [self.lcg_cmd, self.lcg_fb, self.vbs_cmd, self.vbs_fb, self.thruster1_fb, self.thruster2_fb, self.odom]
        temp_list = [self.lcg_fb, self.vbs_fb, self.odom]

        # Set up the ApproximateTimeSynchronizer
        self.synched_message = ApproximateTimeSynchronizer(
            temp_list,
            queue_size = 100,  # How long we wait
            slop = 0.1, # Max time difference, seems like we can make this rather small for SAM and still get a lot of data
            allow_headerless=True
        )

        self.synched_pub = self.create_publisher(SynchedData, "/synched_data", 10)
        self.synched_message.registerCallback(self.callback)

    def callback(self, lcg_fb, vbs_fb, odom): #self, lcg_cmd, lcg_fb, vbs_cmd, vbs_fb, thruster1_fb, thruster2_fb, odom):
        
        # Making message
        sync_msg = SynchedData()
        sync_msg.lcg_fb = lcg_fb
        # sync_msg.lcg_cmd = self.lcg_cmd_msg # lcg_cmd
        sync_msg.odom_gt = odom
        # sync_msg.thruster1_fb = self.thruster1_fb_msg # thruster1_fb
        # sync_msg.thruster1_cmd = self.thruster1_cmd_msg
        # sync_msg.thruster2_fb = self.thruster2_fb_msg # thruster2_fb
        # sync_msg.thruster2_cmd = self.thruster2_cmd_msg
        sync_msg.vbs_fb = vbs_fb
        # sync_msg.vbs_cmd = self.vbs_cmd_msg #vbs_cmd

        # Publish message
        self.synched_pub.publish(sync_msg)
        #self.get_logger().info("Published synched data")

    def thruster1_cmd_cb(self, msg):
        self.thruster1_cmd_msg = msg
    def thruster2_cmd_cb(self, msg):
        self.thruster2_cmd_msg = msg
    def lcg_cmd_cb(self, msg):
        self.lcg_cmd_msg = msg
    def vbs_cmd_cb(self, msg):
        self.vbs_cmd_msg = msg
    def thruster1_fb_cb(self, msg):
        self.thruster1_fb_msg = msg
    def thruster2_fb_cb(self, msg):
        self.thruster1_fb_msg = msg
    

def main(args=None):
    # Start and run node
    rclpy.init(args=args)

    node_stamp = AddTimestamp()
    node_synch = SyncSubscriber()

    executor = MultiThreadedExecutor()
    executor.add_node(node_stamp)
    executor.add_node(node_synch)
    

    executor.spin()

    node_stamp.destroy_node()
    node_synch.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()