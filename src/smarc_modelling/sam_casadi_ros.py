import rclpy, sys, time
from rclpy.node import Node


from tf2_geometry_msgs import PoseWithCovarianceStamped
import tf2_geometry_msgs.tf2_geometry_msgs
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Imu

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from smarc_msgs.msg import PercentStamped, ThrusterRPM, ThrusterFeedback
from smarc_control_msgs.msg import Topics as ControlTopics
from sam_msgs.msg import Topics as SamTopics
from sam_msgs.msg import ThrusterAngles, ThrusterRPMs
from dead_reckoning_msgs.msg import Topics as DRTopics
from rclpy.executors import MultiThreadedExecutor

import numpy as np
from smarc_modelling.vehicles import *
from smarc_modelling.lib import *
from smarc_modelling.vehicles.SAM import SAM
from smarc_modelling.vehicles.SAM_casadi import SAM_casadi

# Initial conditions
eta0 = np.zeros(7)
eta0[2] = 0
eta0[3] = 1.0  # Initial quaternion (no rotation) 
nu0 = np.zeros(6)  # Zero initial velocities
u0 = np.zeros(6)
u0[0] = 50
u0[1] = 50 #45
x0 = np.concatenate([eta0, nu0, u0])

# Simulation timespan
dt = 0.01 #0.01 
t_span = (0, 10)  # 20 seconds simulation
n_sim = int(t_span[1]/dt)
t_eval = np.linspace(t_span[0], t_span[1], n_sim)



class Sol():
    """
    Solver data class to match with Omid's plotting functions
    """
    def __init__(self, t, data) -> None:
        self.t = t
        self.y = data

class RosReplay(Node):

    def __init__(self, node) -> None:

        self._node = node

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        # We need to declare the parameter we want to read out from the alunch file first.
        self._node.declare_parameter('robot_name', 'sam0')
        self._node.declare_parameter('tf_suffix', '')
        self._node.declare_parameter('acados_dir', '')
        tf_suffix = self._node.get_parameter('tf_suffix').get_parameter_value().string_value
        self.robot_name = self._node.get_parameter('robot_name').get_parameter_value().string_value
        self._robot_base_link = self.robot_name + '/base_link'+ tf_suffix
        self._odom_link = self.robot_name + '/odom'
        self.acados_dir = self._node.get_parameter('acados_dir').get_parameter_value().string_value

        self._loginfo(f"robot base link: {self._robot_base_link}")

        self.state_sub = node.create_subscription(msg_type=Odometry, topic=ControlTopics.STATES, callback=self._states_cb, qos_profile=10)

        self.lcg_fb = node.create_subscription(msg_type=PercentStamped, topic=SamTopics.LCG_FB_TOPIC, callback=self._lcg_cb, qos_profile=10)
        self.vbs_fb = node.create_subscription(msg_type=PercentStamped, topic=SamTopics.VBS_FB_TOPIC, callback=self._vbs_cb, qos_profile=10)
        # self.rpm1_fb = node.create_subscription(msg_type=ThrusterRPM, topic=SamTopics.THRUSTER1_CMD_TOPIC, callback=self._lcg_cb, qos_profile=10)
        # self.rpm2_fb = node.create_subscription(msg_type=ThrusterRPM, topic=SamTopics.THRUSTER2_CMD_TOPIC, callback=self._lcg_cb, qos_profile=10)
        # FIXME: Why is this hardcoded?
        self.combined_rpms_fb = node.create_subscription(msg_type=ThrusterRPMs, topic="core/thruster_rpms_cmd", callback=self._rpms_cb, qos_profile=10)
        self.thrust_vector_fb = node.create_subscription(msg_type=ThrusterAngles, topic=SamTopics.THRUST_VECTOR_CMD_TOPIC, callback=self._thrust_vector_cb, qos_profile=10)

        self._waypoint_global = None

        self._tf_base_link_global = None
        self._tf_mocap_base_link = None
        self._tf_odom_global = None

        self._states = Odometry()
        self._received_states = False

        self._states_in_mocap = Odometry()
        self._transformed_state_to_mocap = False

        # Create SAM instance
        self.sam = SAM_casadi()
        self.dynamics = self.sam.dynamics()

        self.dt = 0.1

        self._control_input = {} 
        self._control_input['vbs'] = 0.0
        self._control_input['lcg'] = 0.0
        self._control_input['rpm1'] = 0.0
        self._control_input['rpm2'] = 0.0
        self._control_input['stern'] = 0.0
        self._control_input['rudder'] = 0.0

        # Internal methods
    def _loginfo(self, s):
        self._node.get_logger().info(s)


    def _states_cb(self, msg):
        self._states = msg
        self._received_states = True

    # Control input callbacks added for testing
    def _vbs_cb(self, vbs_fb_msg: PercentStamped):
        self._control_input['vbs'] = vbs_fb_msg.value

    def _lcg_cb(self, lcg_fb_msg: PercentStamped):
        self._control_input['lcg'] = lcg_fb_msg.value

    def _rpms_cb(self, combined_rpms_fb: ThrusterRPMs):
        self._control_input['rpm1'] = combined_rpms_fb.thruster_1_rpm
        self._control_input['rpm2'] = combined_rpms_fb.thruster_2_rpm 

    def _thrust_vector_cb(self, thrust_vector_fb_msg: ThrusterAngles):
        self._control_input['stern'] = thrust_vector_fb_msg.thruster_vertical_radians
        self._control_input['rudder'] = thrust_vector_fb_msg.thruster_horizontal_radians

    def _update_tf(self):
        # FIXME: This could be an issue? What if we do path planning? Which frames do we use then?
        if self._waypoint_global is None:
            return

        try:
            self._tf_base_link_global = self._tf_buffer.lookup_transform(self._robot_base_link,
                                                                  self._waypoint_global.header.frame_id,
                                                                  rclpy.time.Time(seconds=0))
        except Exception as ex:
            self._loginfo(
                f"Could not transform {self._robot_base_link} to {self._waypoint_global.header.frame_id}: {ex}")
            return


        try:
            self._tf_mocap_base_link = self._tf_buffer.lookup_transform(self._waypoint_global.header.frame_id,
                                                                  self._robot_base_link,
                                                                  rclpy.time.Time(seconds=0))
        except Exception as ex:
            self._loginfo(
                f"Could not transform {self._robot_base_link} to {self._waypoint_global.header.frame_id}: {ex}")
            return


        try:
            self._tf_odom_global = self._tf_buffer.lookup_transform(self._odom_link,
                                                                  self._waypoint_global.header.frame_id,
                                                                  rclpy.time.Time(seconds=0))
        except Exception as ex:
            self._loginfo(
                f"Could not transform {self._robot_base_link} to {self._waypoint_global.header.frame_id}: {ex}")
            return

        try:
            self._tf_global_odom = self._tf_buffer.lookup_transform(self._waypoint_global.header.frame_id,
                                                                    self._odom_link,
                                                                  rclpy.time.Time(seconds=0))
        except Exception as ex:
            self._loginfo(
                f"Could not transform {self._robot_base_link} to {self._waypoint_global.header.frame_id}: {ex}")
            return

    def _transform_state(self):
        #self._loginfo(f"states_in_mocap: {self._states_in_mocap}")
        # FIXME: This is never true, because we initialize self._states as Odometry...
        if self._states is None:
            return

        if self._tf_mocap_base_link is None:
            return

        self._states_in_mocap.header.frame_id = self._waypoint_global.header.frame_id
        self._states_in_mocap.pose.pose = tf2_geometry_msgs.do_transform_pose(self._states.pose.pose, self._tf_global_odom)

        self._transformed_state_to_mocap = True

    def rk4(x, u, dt, fun):
        k1 = fun(x, u)
        k2 = fun(x+dt/2*k1, u)
        k3 = fun(x+dt/2*k2, u)
        k4 = fun(x+dt*k3, u)

        x_t = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        np.set_printoptions(precision=3)
        print(f"vbs: {x_t.full().flatten()[13]}; vbs_dot: {k1.full().flatten()[13]}")

        return x_t.full().flatten()

    def run_simulation(self):
        """
        Run SAM simulation using solve_ivp.
        """

        if not self.initialized:
            return

        u = np.zeros(6)
        u[0] = self._control_input['vbs']
        u[1] = self._control_input['lcg']
        u[2] = self._control_input['stern']
        u[3] = self._control_input['rudder']
        u[4] = self._control_input['rpm1']
        u[5] = self._control_input['rpm2']

        sol = rk4(data[:,i], u, self.dt, self.dynamics)


