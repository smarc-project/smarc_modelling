#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Contains various functions that are used multiple times across the different PIML files

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from smarc_modelling.lib.gnc import *


def load_rosbag(bag_path: str=""):
    """Loads the data from a rosbag and separates it out into the different state vectors"""

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
    dS = []
    dR = []
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
    q0 = []
    q1 = []
    q2 = []
    q3 = []
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
        time.append(msg.lcg_cmd.header.stamp.sec + msg.lcg_cmd.header.stamp.nanosec * 1e-9) # Time from lcg_cmd but all headers are synched so really does not matter
        
        # Controls
        lcg_fb.append(msg.lcg_fb.value)
        lcg_cmd.append(msg.lcg_cmd.value)
        dS.append(msg.thrust_vector_cmd.thruster_vertical_radians)
        dR.append(msg.thrust_vector_cmd.thruster_horizontal_radians)
        rpm1_fb.append(msg.thruster1_fb.rpm.rpm)
        rpm1_cmd.append(msg.thruster1_cmd.rpm) 
        rpm2_fb.append(msg.thruster2_fb.rpm.rpm)
        rpm2_cmd.append(msg.thruster2_cmd.rpm)
        vbs_fb.append(msg.vbs_fb.value)
        vbs_cmd.append(msg.vbs_cmd.value)

        # Pose
        x.append(msg.odom_gt.pose.pose.position.x)
        y.append(msg.odom_gt.pose.pose.position.y)
        z.append(msg.odom_gt.pose.pose.position.z)
        q0.append(msg.odom_gt.pose.pose.orientation.w)
        q1.append(msg.odom_gt.pose.pose.orientation.x)
        q2.append(msg.odom_gt.pose.pose.orientation.y)
        q3.append(msg.odom_gt.pose.pose.orientation.z)

        # Speeds
        u.append(msg.odom_gt.twist.twist.linear.x)
        v.append(msg.odom_gt.twist.twist.linear.y)
        w.append(msg.odom_gt.twist.twist.linear.z)
        p.append(msg.odom_gt.twist.twist.angular.x)
        q.append(msg.odom_gt.twist.twist.angular.y)
        r.append(msg.odom_gt.twist.twist.angular.z)

    # Calculating acceleration numerically (No ROS source)
    u_dot = np.gradient(u, time)
    v_dot = np.gradient(v, time)
    w_dot = np.gradient(w, time)
    p_dot = np.gradient(p, time)
    q_dot = np.gradient(q, time)
    r_dot = np.gradient(r, time)
    
    eta = [x, y, z, q0, q1, q2, q3]
    nu = [u, v, w, p, q, r]
    acc = [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
    u_control = [vbs_cmd, lcg_cmd, dS, dR, rpm1_cmd, rpm2_cmd]
    u_control_ref = [vbs_fb, lcg_fb, dS, dR, rpm1_cmd, rpm2_cmd] # We use rpm_cmd here since rpm_fb is not working atm
    
    return time, eta, nu, acc, u_control, u_control_ref


def load_data_from_bag(data_file: str="", return_type: str=""):
    """Loads the rosbag and does the needed post processing to calculate things needed for physics loss"""

    # Importing SAM for getting some values from dynamics function
    from smarc_modelling.vehicles.SAM import SAM

    # Loading bag
    time, eta, nu, acc, u_cmd, u_fb = load_rosbag(data_file)

    # Transposing data
    time, eta, nu, acc, u_cmd, u_fb = [np.transpose(x) for x in (time, eta, nu, acc, u_cmd, u_fb)]
    state_vector = np.concatenate([eta, nu, u_fb], axis=1) 

    # SAM white-box model to get M, C and g
    dt_vec = np.diff(time)
    sam = SAM(dt_vec[0])

    # Things needed for physics loss
    Dv_comp = np.zeros((len(state_vector), 6))
    Mv_dot = np.zeros_like(nu)
    Cv = np.zeros_like(nu)
    g_eta = np.zeros_like(nu)
    tau = np.zeros_like(nu)
    M = []

    for t in range(len(state_vector)):

        # Computing acceleration if D = 0
        try:
            sam.update_dt(dt_vec[t])
        except:
            # dt vec is one shorter than the rest of the data since we used diff for it, so last value is just the mean time-step of the set
            dt_mean = np.mean(dt_vec)
            sam.update_dt(dt_mean)
        sam.dynamics(state_vector[t], u_cmd[t]) # Calling the dynamics to update all matrices directly

        # Calculated acceleration where we have no damping
        v_dot_nod = sam.Minv @ (sam.tau - sam.C @ nu[t] - sam.g_vec)

        # Calculate the damping force based on difference in model prediction and real data
        Dv_comp[t] = sam.M @ (v_dot_nod.T - acc[t].T)

        # Calculating miscellaneous things for physics loss
        Mv_dot[t] = sam.M @ acc[t]
        Cv[t] = sam.C @ nu[t]
        g_eta[t] = sam.g_vec
        tau[t] = sam.tau
        M.append(sam.M)

    time = time - time[0] # Setting time to start at 0

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
            torch.tensor(time, dtype=torch.float32),
            torch.tensor(np.array(M), dtype=torch.float32),
            torch.tensor(acc, dtype=torch.float32)
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
            time,
            M,
            acc
        )
    

def load_to_trajectory(data_files: list):
    
    x_trajectories = []
    y_trajectories = []

    for dataset in data_files:
        # Load data from bag
        path = "src/smarc_modelling/piml/data/rosbags/" + dataset
        eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, acc = load_data_from_bag(path, "torch")
        
        x_traj = torch.cat([nu, u], dim=1)
        y_traj = {
            "eta": eta,
            "u_cmd": u_cmd,
            "Dv_comp": Dv_comp,
            "Mv_dot": Mv_dot,
            "Cv": Cv,
            "g_eta": g_eta,
            "tau": tau,
            "t": t,
            "nu": nu,
            "M": M,
            "acc": acc,
            "u": u
        }

        # Append most recently loaded data
        x_trajectories.append(x_traj)
        y_trajectories.append(y_traj)

    return x_trajectories, y_trajectories


def eta_quat_to_rad(eta, return_type="numpy"):
    """Turns quaternion in eta to radians"""
    pose = eta[0:3]
    quat = eta[3:]
    euler = R.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=False)
    if return_type == "numpy":
        return np.hstack([pose, euler])
    if return_type == "torch":
        pose = torch.tensor(pose, dtype=torch.float32)
        euler = torch.tensor(euler, dtype=torch.float32)
        return torch.cat([pose, euler])


def eta_quat_to_deg(eta):
    """Turns quaternion in eta to angles"""
    eta = eta_quat_to_rad(eta)
    eta[3] *= (180/np.pi)
    eta[4] *= (180/np.pi)
    eta[5] *= (180/np.pi)
    return eta


def angular_vel_to_quat_vel(eta, nu):
    """Calculate angular velocity in degrees to quaternion speeds"""

    # Takes an angular velocity and returns it as a quaternion velocity
    angle = eta[3:]
    q = R.from_euler("xyz", angle).as_quat(scalar_first=True)
    q = q/np.linalg.norm(q)

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
