#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Contains various functions that are used multiple times across the different PIML model files

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import pandas as pd
from smarc_modelling.lib.gnc import *

def load_rosbag(bag_path):
    # Initialize reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get all topics and type
    topics = reader.get_all_topics_and_types()

    # Find message type
    topic_type_map = {topic.name: topic.type for topic in topics}
    msg_class = get_message(topic_type_map.get("/synched_data"))
    
    # Data for output
    time = []
    # Controls
    lcg_cmd = []
    lcg_fb = []
    dR = []
    dS = []
    rpm1_cmd = []
    rpm1_fb = []
    rpm2_cmd = []
    rpm2_fb = []
    vbs_cmd = []
    vbs_fb = []
    # Pose
    x = []
    y = []
    z = []
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    # Speeds
    u = []
    v = []
    w = []
    p = []
    q = []
    r = []
    # Accelerations
    u_dot = []
    v_dot = []
    w_dot = []
    p_dot = []
    q_dot = []
    r_dot = []
    
    # Read messages
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        # Getting msg into right format
        msg = deserialize_message(data, msg_class)

        # Appending data to vectors
        # Time
        time.append(msg.imu.header.stamp.sec + msg.imu.header.stamp.nanosec * 1e-9) # Time from imu but all headers are synched
        
        # Controls
        lcg_fb.append(msg.lcg_fb.value)
        lcg_cmd = lcg_fb # No cmd in bag atm
        # Thrust vector
        # No cmd in bag atm
        rpm1_fb.append(msg.thruster1_fb.rpm.rpm)
        rpm1_cmd = rpm1_fb 
        rpm2_fb.append(msg.thruster2_fb.rpm.rpm)
        rpm2_cmd = rpm2_fb
        vbs_fb.append(msg.vbs_fb.value)
        vbs_cmd = vbs_fb

        # Pose
        x.append(msg.odom_gt.pose.pose.position.x)
        y.append(msg.odom_gt.pose.pose.position.y)
        z.append(msg.odom_gt.pose.pose.position.z)
        q1.append(msg.odom_gt.pose.pose.orientation.x)
        q2.append(msg.odom_gt.pose.pose.orientation.y)
        q3.append(msg.odom_gt.pose.pose.orientation.z)
        q4.append(msg.odom_gt.pose.pose.orientation.w)

        # Speeds
        u.append(msg.odom_gt.twist.twist.linear.x)
        v.append(msg.odom_gt.twist.twist.linear.y)
        w.append(msg.odom_gt.twist.twist.linear.z)
        p.append(msg.odom_gt.twist.twist.angular.x)
        q.append(msg.odom_gt.twist.twist.angular.y)
        r.append(msg.odom_gt.twist.twist.angular.z)

        # Accelerations
        u_dot.append(msg.imu.linear_acceleration.x)
        v_dot.append(msg.imu.linear_acceleration.y)
        w_dot.append(msg.imu.linear_acceleration.z)
    
    # Calculating angular acceleration numerically (No ROS source)
    p_dot = np.gradient(p, time)
    q_dot = np.gradient(q, time)
    r_dot = np.gradient(r, time)
    
    eta = [x, y, z, q1, q2, q3, q4]
    nu = [u, v, w, p, q, r]
    acc = [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
    dS = np.zeros_like(lcg_cmd)
    dR = dS
    u_control = [vbs_cmd, lcg_cmd, dS, dR, rpm1_cmd, rpm2_cmd]
    u_control_ref = [vbs_fb, lcg_fb, dS, dR, rpm1_fb, rpm2_fb]
    
    return time, eta, nu, acc, u_control, u_control_ref


def load_rosbag_brov(bag_path):
    # Same as load_rosbag but is structured for bag from BlueROV instead of SAM

    # Initialize reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get all topics and type
    topics = reader.get_all_topics_and_types()
    # Get message class type
    topic_type_map = {topic.name: topic.type for topic in topics}
    target_topic = "/synced_pose_control_rcout"
    msg_type = topic_type_map[target_topic]
    msg_class = get_message(msg_type)

    # Data for output
    time = []
    # Controls commands
    cmd_1 = []
    cmd_2 = []
    cmd_3 = []
    cmd_4 = []
    cmd_5 = []
    cmd_6 = []
    cmd_7 = []
    cmd_8 = []
    # Controls feedback
    fb_1 = []
    fb_2 = []
    fb_3 = []
    fb_4 = []
    fb_5 = []
    fb_6 = []
    fb_7 = []
    fb_8 = []
    # Pose
    x = []
    y = []
    z = []
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    # Speed
    u = []
    v = []
    w = []
    p = []
    q = []
    r = []

    # BlueROV specific stuff
    found_first_action = False
    once = True
    # Read messages
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()

        if topic != target_topic:
            continue # Bypass /tf messages

        # Getting msg into right format
        msg = deserialize_message(data, msg_class)
        
        # Checking for first control message
        channels_cmd = list(msg.rc_stamped.rc.channels)[:8] # Channels 7, 8 are not good here
        if not found_first_action:
            # Check if all 8 channels are zero
            if all(ch == 1500 for ch in channels_cmd):
                continue # still in the skip phase
            else:
                found_first_action = True
        channels_fb = list(msg.rc_out.channels)[:8]
 
        # Appending data to vectors
        # Time
        time.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        # Pose
        x.append(msg.pose.pose.position.x)
        y.append(msg.pose.pose.position.y)
        z.append(msg.pose.pose.position.z)
        q1.append(msg.pose.pose.orientation.x)
        q2.append(msg.pose.pose.orientation.y)
        q3.append(msg.pose.pose.orientation.z)
        q4.append(msg.pose.pose.orientation.w)

        # Speeds
        u.append(msg.twist.twist.linear.x)
        v.append(msg.twist.twist.linear.y)
        w.append(msg.twist.twist.linear.z)
        p.append(msg.twist.twist.angular.x)
        q.append(msg.twist.twist.angular.y)
        r.append(msg.twist.twist.angular.z)

        # Controls commands
        cmd_1.append(channels_cmd[0])
        cmd_2.append(channels_cmd[1])
        cmd_3.append(channels_cmd[2])
        cmd_4.append(channels_cmd[3])
        cmd_5.append(channels_cmd[4])
        cmd_6.append(channels_cmd[5])
        cmd_7.append(channels_cmd[6])
        cmd_8.append(channels_cmd[7])

        # Controls feedback
        fb_1.append(channels_fb[0])
        fb_2.append(channels_fb[1])
        fb_3.append(channels_fb[2])
        fb_4.append(channels_fb[3])
        fb_5.append(channels_fb[4])
        fb_6.append(channels_fb[5])
        fb_7.append(channels_fb[6])
        fb_8.append(channels_fb[7])

    # Calculating accelerations from speeds
    u_dot = np.gradient(u, time)
    v_dot = np.gradient(v, time)
    w_dot = np.gradient(w, time)
    p_dot = np.gradient(p, time)
    q_dot = np.gradient(q, time)
    r_dot = np.gradient(r, time)

    eta = [x, y, z, q1, q2, q3, q4]
    nu = [u, v, w, p, q, r]
    acc = [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
    u_control = [cmd_1, cmd_2, cmd_3, cmd_4, cmd_5, cmd_6, cmd_7, cmd_8]
    u_control_ref = [fb_1, fb_2, fb_3, fb_4, fb_5, fb_6, fb_7, fb_8]

    return time, eta, nu, acc, u_control, u_control_ref


def load_data_from_bag(data_file: str="", return_type: str=""):
    from smarc_modelling.vehicles.SAM import SAM

    # Processing data into right format from ROS bag
    print(f" Getting data from ROS bag...")
    time, eta, nu, acc, u_cmd, u_fb = load_rosbag(data_file)
    
    # Need to transpose everything here to fit previous code...
    time, eta, nu, acc, u_cmd, u_fb = [np.transpose(x) for x in (time, eta, nu, acc, u_cmd, u_fb)]
    state_vector = np.concatenate([eta, nu, u_fb], axis=1)

    # SAM white-box model to create M, C and g
    dt = np.mean(np.diff(time)) # Getting a dt as mean of time step difference
    sam = SAM(dt)

    # Pre-allocate space for different vectors and matrices
    v_dot_ord = np.zeros_like(nu)
    v_dot_nod = np.zeros_like(nu)
    Dv_comp = np.zeros((len(state_vector), 6))

    # These will be needed when computing the physics loss
    Mv_dot = np.zeros_like(nu)
    Cv = np.zeros_like(nu)
    g_eta = np.zeros_like(nu)
    tau = np.zeros_like(nu)

    print(f" Calculating D(v)v...")

    for t in range(len(state_vector)):

        # Full acceleration from irl
        v_dot_ord[t] = acc[t]

        # Computing model prediction where D = 0
        # Calling SAM dynamics to update model matrices
        sam.dynamics(state_vector[t], u_cmd[t]) 
        v_dot_nod[t] = sam.Minv @ (sam.tau - sam.C @ nu[t] - sam.g_vec)
        
        # Calculate the damping force based on difference in model prediction and real data
        Dv_comp[t] = sam.M @ (v_dot_nod[t].T - v_dot_ord[t].T)

        # Calculating miscellaneous things for physics loss
        Mv_dot[t] = sam.M @ v_dot_ord[t]
        Cv[t] = sam.C @ nu[t]
        g_eta[t] = sam.g_vec
        tau[t] = sam.tau

    print(f" Data has been loaded and processed!")
    if return_type == "torch": # Return all values as torch tensors
        return(
            torch.tensor(eta, dtype=torch.float32),
            torch.tensor(nu, dtype=torch.float32),
            torch.tensor(u_fb, dtype=torch.float32),
            torch.tensor(u_cmd, dtype=torch.float32),
            torch.tensor(Dv_comp, dtype=torch.float32),
            torch.tensor(Mv_dot, dtype=torch.float32),
            torch.tensor(Cv, dtype=torch.float32),
            torch.tensor(g_eta, dtype=torch.float32),
            torch.tensor(tau, dtype=torch.float32),
            torch.tensor(time, dtype=torch.float32)
        )
    else: # Return all values as numpy matrices
        return(
            eta,
            nu,
            u_fb,
            u_cmd,
            Dv_comp,
            Mv_dot,
            Cv,
            g_eta,
            tau,
            time
        )


def load_data_from_bag_brov(data_file: str="", return_type: str=""):
    from smarc_modelling.vehicles.BlueROV_PIML import BlueROV_PIML

    # Processing data into right format from ROS bag
    time, eta, nu, acc, u_cmd, u_fb = load_rosbag_brov(data_file)
    
    # Need to transpose everything here to fit previous code...
    time, eta, nu, acc, u_cmd, u_fb = [np.transpose(x) for x in (time, eta, nu, acc, u_cmd, u_fb)]
    state_vector = np.concatenate([eta, nu, u_fb], axis=1)

    # BlueROV white-box model to create M, C and g
    dt = np.mean(np.diff(time)) # Getting a dt as mean of time step difference
    brov = BlueROV_PIML(h=dt)

    # Pre-allocate space for different vectors and matrices
    v_dot_ord = np.zeros_like(nu)
    v_dot_nod = np.zeros_like(nu)
    Dv_comp = np.zeros((len(state_vector), 6))

    # These will be needed when computing physics loss
    Mv_dot = np.zeros_like(nu)
    Cv = np.zeros_like(nu)
    g_eta = np.zeros_like(nu)
    tau = np.zeros_like(nu)

    print(f" Calculating D(v)v...")

    for t in range(len(state_vector)):

        # Full acceleration from irl
        v_dot_ord[t] = acc[t]

        # Get and M and C from model
        M = brov.create_M()
        C = brov.create_C(nu[t])
 
        # Calculate the damping force based on difference in model prediction and real data
        Dv_comp[t] = M @ (v_dot_nod[t].T - v_dot_ord[t].T)

        # Calculating miscellaneous things for physics loss
        Mv_dot[t] = M @ v_dot_ord[t]
        Cv[t] = C @ nu[t]
        g_eta[t] = brov.create_g(eta[t]).squeeze()
        tau[t] = brov.create_F(u_fb[t]).squeeze()


    print(f" Data has been loaded and processed!")
    if return_type == "torch": # Return all values as torch tensors
        return(
            torch.tensor(eta, dtype=torch.float32),
            torch.tensor(nu, dtype=torch.float32),
            torch.tensor(u_fb, dtype=torch.float32),
            torch.tensor(u_cmd, dtype=torch.float32),
            torch.tensor(Dv_comp, dtype=torch.float32),
            torch.tensor(Mv_dot, dtype=torch.float32),
            torch.tensor(Cv, dtype=torch.float32),
            torch.tensor(g_eta, dtype=torch.float32),
            torch.tensor(tau, dtype=torch.float32),
            torch.tensor(time, dtype=torch.float32)
        )
    else: # Return all values as numpy matrices
        return(
            eta,
            nu,
            u_fb,
            u_cmd,
            Dv_comp,
            Mv_dot,
            Cv,
            g_eta,
            tau,
            time
        )
        

def load_data(data_file: str = "", return_type: str=""):
    from smarc_modelling.vehicles.SAM import SAM

    # NOTE: This will later come baked into the data file
    u_ref = np.zeros(6)
    u_ref[0] = 50               # VBS
    u_ref[1] = 50               # LCG
    u_ref[2] = np.deg2rad(7)    # Vertical (stern)
    u_ref[3] = -np.deg2rad(7)   # Horizontal (rudder)
    u_ref[4] = 1000             # RPM 1
    u_ref[5] = u_ref[4]         # RPM 2

    # Load the data file
    print(f" Loading data...")
    data = pd.read_csv(data_file)
    # Processing
    time = data["Time"].values
    eta = data[["x", "y", "z", "q0", "q1", "q2", "q3"]].values
    nu = data[["u", "v", "w", "p", "q", "r"]].values
    u = data[["VBS", "LCG", "DS", "DR", "RPM1", "RPM2"]].values
    state_vector = np.concatenate([eta, nu, u], axis=1)

    # Initialize SAM white-box model
    # TODO: Get dt from the time data
    dt = 0.01
    sam = SAM(dt)

    # Pre-allocate space for different vectors and matrices
    v_dot_ord = np.zeros_like(nu)
    v_dot_nod = np.zeros_like(nu)
    Dv_comp = np.zeros((len(state_vector), 6))

    # These will be needed when computing the physics loss
    Mv_dot = np.zeros_like(nu)
    Cv = np.zeros_like(nu)
    g_eta = np.zeros_like(nu)
    tau = np.zeros_like(nu)

    for t in range(len(state_vector)):

        # TODO: Remove when u_ref is in data file
        if t == 3000:
            u_ref[2] = 0
            u_ref[3] = 0

        # Computing the full model prediction
        # TODO: Replace with real acceleration when we have it
        v_dot_ord[t] = sam.dynamics(state_vector[t], u_ref)[7:13]

        # Computing model prediction where D = 0
        # Calling SAM dynamics to update model matrices
        sam.dynamics(state_vector[t], u_ref) 
        v_dot_nod[t] = sam.Minv @ (sam.tau - sam.C @ nu[t] - sam.g_vec)

        # Calculate the damping force based on difference in model prediction and real data
        Dv_comp[t] = sam.M @ (v_dot_nod[t].T - v_dot_ord[t].T)

        # Calculating miscellaneous things for physics loss
        Mv_dot[t] = sam.M @ v_dot_ord[t]
        Cv[t] = sam.C @ nu[t]
        g_eta[t] = sam.g_vec
        tau[t] = sam.tau

    print(f" Data has been loaded!")
    if return_type == "torch": # Return all values as torch tensors
        return(
            torch.tensor(eta, dtype=torch.float32),
            torch.tensor(nu, dtype=torch.float32),
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(Dv_comp, dtype=torch.float32),
            torch.tensor(Mv_dot, dtype=torch.float32),
            torch.tensor(Cv, dtype=torch.float32),
            torch.tensor(g_eta, dtype=torch.float32),
            torch.tensor(tau, dtype=torch.float32),
            torch.tensor(time, dtype=torch.float32)
        )
    else: # Return all values as numpy matrices
        return(
            eta,
            nu,
            u,
            Dv_comp,
            Mv_dot,
            Cv,
            g_eta,
            tau,
            time
        )
    
def eta_quat_to_deg(eta):
    pose = eta[0:3]
    quat = eta[3:]
    euler = R.from_quat(quat).as_euler("xyz", degrees=False)
    return np.hstack([pose, euler])

def angular_vel_to_quat_vel(eta, nu):

    # Takes an angular velocity and returns it as a quaternion velocity
    angle = eta[3:]
    q = R.from_euler("xyz", angle).as_quat()
    q = q/np.linalg.norm(q)

    # Convert quaternion to DCM for position kinematics
    C = quaternion_to_dcm(q)

    # Position dynamics: ṗ = C * v
    pos_dot = nu[0:3]

    ## From Fossen 2021, eq. 2.78:
    om = nu[3:6]  # Angular velocity
    q0, q1, q2, q3 = q
    T_q_n_b = 0.5 * np.array([
                                [-q1, -q2, -q3],
                                [q0, -q3, q2],
                                [q3, q0, -q1],
                                [-q2, q1, q0]
                                ])
    q_dot = T_q_n_b @ om + 100/2 * (1 - q.T.dot(q)) * q

    return np.concatenate([pos_dot, q_dot])
