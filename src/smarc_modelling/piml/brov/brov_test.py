#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smarc_modelling.piml.utils.utility_functions import load_data_from_bag_brov
from smarc_modelling.piml.piml_simulator import VEHICLE_SIM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("Testing BROV functions")

    # Directory to data
    bag_name = "src/smarc_modelling/piml/data/brovbags/Bag_Tank_3"
    
    print("Loading bag...")
    eta, nu, u_fb, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, time = load_data_from_bag_brov(bag_name, "")
    init_pose = np.concatenate([eta[0], nu[0], u_fb[0]])
    
    # All neutral
    u_fb_empty = np.array([1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]) # All inputs on "0"

    # Full force directional inputs
    u_fb_forward = np.array([1900, 1900, 1100, 1100, 1500, 1500, 1500, 1500]) # Full force forward --> 85.98 Good
    u_fb_back = np.array([1100, 1100, 1900, 1900, 1500, 1500, 1500, 1500]) # Full force back --> -85.98 Good

    u_fb_right_strafe = np.array([1900, 1100, 1900, 1100, 1500, 1500, 1500, 1500]) # Full force strafing right --> -85.98 Good
    u_fb_left_strafe = np.array([1100, 1900, 1100, 1900, 1500, 1500, 1500, 1500]) # Full force strafing right --> 85.98 Good

    u_fb_heave_down = np.array([1500, 1500, 1500, 1500, 1900, 1100, 1100, 1900]) # Full force down --> -121.6 Good
    u_fb_heave_up = np.array([1500, 1500, 1500, 1500, 1100, 1900, 1900, 1100]) # Full force up --> 121.6 Good

    u_fb_roll_right = np.array([1500, 1500, 1500, 1500, 1900, 1900, 1100, 1100]) # Full roll right --> 26.5 Good
    u_fb_roll_left = np.array([1500, 1500, 1500, 1500, 1100, 1100, 1900, 1900]) # Full roll left --> -26.5 Good

    u_fb_heave_right = np.array([1500, 1500, 1500, 1500, 1100, 1900, 1100, 1900]) # Full heave right --> -14.59 Good
    u_fb_heave_left = np.array([1500, 1500, 1500, 1500, 1900, 1100, 1900, 1100]) # Full heave left --> 14.59 Good

    u_fb_yaw_right = np.array([1100, 1900, 1900, 1100, 1500, 1500, 1500, 1500]) # Full yaw right --> 22.95 Good
    u_fb_yaw_left = np.array([1900, 1100, 1100, 1900, 1500, 1500, 1500, 1500]) # Full yaw right --> -22.95 Good

    # What controls to use
    u_fb_custom = u_fb_back
    u_fb_custom = np.tile(u_fb_custom, (125, 1))

    print("Initialize model...")
    brov_sim = VEHICLE_SIM("pinn", 0.1, init_pose, time, u_fb_custom, "BROV")
    results = brov_sim.run_sim()[0:7, :].T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for vector, label in zip([eta, results], ["Ground Truth", "Sim Results"]):
        # Plotting trajectories
        points = vector[:, :3].T
        # Attaching 3D axis to the figure
        ax.plot(points[0], points[1], -points[2], label=label)

    # Settings for plot
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.legend()
    plt.show()
