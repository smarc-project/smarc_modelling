#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For do_transform_twist
from scipy.spatial.transform import Rotation as R
import numpy as np

# ROS imports
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import Buffer, TransformListener

# Message types
from smarc_msgs.msg import PercentStamped, ThrusterRPM, ThrusterFeedback
from sam_msgs.msg import ThrusterAngles
from piml_msgs.msg import ThrusterRPMStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, Twist

class AddTimestamp(Node):
    """Node to add new timestamps to bags, simply subs then publishes with new stamps from clock time directly on received message"""

    def __init__(self):
        super().__init__('add_timestamps')

        # --< Transformers >-- #
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --< Subscribers >-- #
        # Thrusters
        self.thruster1_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster1_cmd", self.add_stamp_thruster1, 1) # No stamp at all
        self.thruster2_cmd_sub = self.create_subscription(ThrusterRPM, "/sam/core/thruster2_cmd", self.add_stamp_thruster2, 1) # No stamp at all
        self.thruster1_fb_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster1_fb", self.add_stamp_thruster1fb, 1) # No data in stamp 
        self.thruster2_fb_sub = self.create_subscription(ThrusterFeedback, "/sam/core/thruster2_fb", self.add_stamp_thruster2fb, 1) # No data in stamp
        self.thrust_vectoring_sub = self.create_subscription(ThrusterAngles, "/sam/core/thrust_vector_cmd", self.add_stamp_vector, 1) # Has data in stamp but need to reassign
        # LCG & VBS
        self.lcg_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/lcg_cmd", self.add_stamp_lcg_cmd, 1) # No data in stamp
        self.lcg_fb_sub = self.create_subscription(PercentStamped, "/sam/core/lcg_fb", self.add_stamp_lcg_fb, 1) # Has data in stamp but need to reassign
        self.vbs_cmd_sub = self.create_subscription(PercentStamped, "/sam/core/vbs_cmd", self.add_stamp_vbs_cmd, 1) # No data in stamp
        self.vbs_fb_sub = self.create_subscription(PercentStamped, "/sam/core/vbs_fb", self.add_stamp_vbs_fb, 1) # Has data in stamp but need to reassign
        # Odom
        # 1970 is /mocap/sam_mocap/odom and 2025 is (typically) /mocap/sam_mocap2/odom
        self.odom_sub = self.create_subscription(Odometry, "/mocap/sam_mocap2/odom", self.add_stamp_odom, 1) # Has data in stamp but need to reassign
        # Velocity 
        # Same as with odom about mocap vs mocap2
        self.velocity_sub = self.create_subscription(TwistStamped, "/mocap/sam_mocap2/velocity", self.add_stamp_velo, 1) # Has data in stamp but need to reassign

        # --< Publishers >-- #
        # Thrusters
        self.thruster1_cmd_pub = self.create_publisher(ThrusterRPMStamped, "/piml/thruster1_cmd", 1)
        self.thruster2_cmd_pub = self.create_publisher(ThrusterRPMStamped, "/piml/thruster2_cmd", 1)
        self.thruster1_fb_pub = self.create_publisher(ThrusterFeedback, "/piml/thruster1_fb", 1) 
        self.thruster2_fb_pub = self.create_publisher(ThrusterFeedback, "/piml/thruster2_fb", 1)
        self.thrust_vectoring_pub = self.create_publisher(ThrusterAngles, "/piml/thrust_vector_cmd", 1)
        # LCG & VBS
        self.lcg_cmd_pub = self.create_publisher(PercentStamped, "/piml/lcg_cmd", 1)
        self.lcg_fb_pub = self.create_publisher(PercentStamped, "/piml/lcg_fb", 1)
        self.vbs_cmd_pub = self.create_publisher(PercentStamped, "/piml/vbs_cmd", 1)
        self.vbs_fb_pub = self.create_publisher(PercentStamped, "/piml/vbs_fb", 1)
        # Odom
        self.odom_pub = self.create_publisher(Odometry, "/piml/odom", 1)
        # Velocity
        self.velo_pub = self.create_publisher(TwistStamped, "/piml/velocity", 1)
        
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

    def add_stamp_velo(self, msg):
        msg_stamped = TwistStamped()
        # The lookup frame is either sam_mocap/base_link or sam_mocap2/base_link dependent on the bag
        transform = self.tf_buffer.lookup_transform("sam_mocap2/base_link", "mocap", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
        msg_twist = msg.twist
        msg_stamped.twist = do_transform_twist(msg_twist, transform)
        msg_stamped.header.stamp = self.get_clock().now().to_msg()
        self.velo_pub.publish(msg_stamped)


def do_transform_twist(twist_msg, transform):
    # For some reason rclpy does not have do_transform_twist... so this is a bit of a workaround for that

    # Get rotation
    q = transform.transform.rotation
    rot = R.from_quat([q.x, q.y, q.z, q.w])

    # Setup vectors to be rotated
    lin = np.array([twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z])
    ang = np.array([twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z])

    # Rotate
    lin_trans = rot.apply(lin)
    ang_trans = rot.apply(ang)

    # Construct new message
    transformed = Twist()
    transformed.linear.x = lin_trans[0]
    transformed.linear.y = lin_trans[1]
    transformed.linear.z = lin_trans[2]
    transformed.angular.x = ang_trans[0]
    transformed.angular.y = ang_trans[1]
    transformed.angular.z = ang_trans[2]

    return transformed


def main():

    # Init ROS
    rclpy.init()
    
    # Create node and add to executor
    timestamp_node = AddTimestamp()
    exec = MultiThreadedExecutor()
    exec.add_node(timestamp_node)

    # Spin the node
    try:
        exec.spin()
    except Exception as e:
        timestamp_node.destroy_node()
        print(e)
        rclpy.shutdown()

    
if __name__ == "__main__":
    main()