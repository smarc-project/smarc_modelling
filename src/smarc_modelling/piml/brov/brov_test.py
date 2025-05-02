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
    print("Initialize model...")
    brov_sim = VEHICLE_SIM(None, 0.1, init_pose, time, u_fb, "BROV")
    results = brov_sim.run_sim()[0:7, :].T
    print(np.shape(results))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for vector, label in zip([eta, results], ["Ground Truth", "Sim Results"]):
        # Plotting trajectories
        points = vector[:10, :3].T
        print(np.shape(points))
        # Attaching 3D axis to the figure
        ax.plot(-points[0], points[1], -points[2], label=label)

    # Settings for plot
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.legend()
    plt.show()
