#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu
from smarc_msgs.msg import PercentStamped, ThrusterRPM, ThrusterFeedback
from nav_msgs.msg import Odometry
from sam_msgs.msg import ThrusterAngles
from piml_msgs.msg import SynchedData

# Compiles a synchronized message from SAM for easy creation of training data
class SyncSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')

        # Create subscribers for different topics
        self.clock = Subscriber(self, Clock, "/clock")
        self.imu = Subscriber(self, Imu, "/sam_auv_v1/core/imu")
        self.lcg_cmd = Subscriber(self, PercentStamped, "/sam_auv_v1/core/lcg_cmd")
        self.lcg_fb = Subscriber(self, PercentStamped, "/sam_auv_v1/core/lcg_fb")
        self.odom_gt = Subscriber(self, Odometry, "/sam_auv_v1/core/odom_gt")
        self.thrust_vector_cmd = Subscriber(self, ThrusterAngles, "/sam_auv_v1/core/thrust_vector_cmd")
        self.thruster1_cmd = Subscriber(self, ThrusterRPM, "/sam_auv_v1/core/thruster1_cmd")
        self.thruster1_fb = Subscriber(self, ThrusterFeedback, "/sam_auv_v1/core/thruster1_fb")
        self.thruster2_cmd = Subscriber(self, ThrusterRPM, "/sam_auv_v1/core/thruster2_cmd")
        self.thruster2_fb = Subscriber(self, ThrusterFeedback, "sam_auv_v1/core/thruster2_fb")
        self.vbs_cmd = Subscriber(self, PercentStamped, "/sam_auv_v1/core/vbs_cmd")
        self.vbs_fb = Subscriber(self, PercentStamped, "/sam_auv_v1/core/vbs_fb")

        # TODO: Sub list is the full list, since we dont publish cmd values atm we have to use a sub-list that contains everything that actually publishes
        sub_list = [self.clock, self.imu, self.lcg_cmd, self.lcg_fb, self.odom_gt, self.thrust_vector_cmd, self.thruster1_cmd, self.thruster1_fb, self.thruster2_cmd, self.thruster2_fb, self.vbs_cmd, self.vbs_fb]
        pub_list = [self.imu, self.lcg_fb, self.odom_gt, self.thruster1_fb, self.thruster2_fb, self.vbs_fb]

        # Set up the ApproximateTimeSynchronizer
        self.synched_message = ApproximateTimeSynchronizer(
            pub_list,
            queue_size = 100,  # How long we wait
            slop = 0.0001  # Max time difference, seems like we can make this rather small for SAM and still get a lot of data
        )

        self.synched_pub = self.create_publisher(SynchedData, "/synched_data", 10)
        self.synched_message.registerCallback(self.callback)

    def callback(self, imu_msg, lcg_fb_msg, odom_gt_msg, thruster1_fb_msg, thruster2_fb_msg, vbs_fb_msg):
        
        # Making message
        sync_msg = SynchedData()
        sync_msg.imu = imu_msg
        sync_msg.lcg_fb = lcg_fb_msg
        sync_msg.odom_gt = odom_gt_msg
        sync_msg.thruster1_fb = thruster1_fb_msg
        sync_msg.thruster2_fb = thruster2_fb_msg
        sync_msg.vbs_fb = vbs_fb_msg

        # Publish message
        self.synched_pub.publish(sync_msg)
        self.get_logger().info("Published synched data")

def main(args=None):
    # Start and run node
    rclpy.init(args=args)
    node = SyncSubscriber()
    rclpy.spin(node)
    # Safe shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()