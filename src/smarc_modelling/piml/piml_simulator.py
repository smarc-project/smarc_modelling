#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Runs euler forward integration to simulate vehicles
# Currently works with
# White-Box models
# PINN models

import numpy as np
from smarc_modelling.vehicles.SAM_PIML import SAM_PIML # Customized SAM using PIML predictions for D
from smarc_modelling.vehicles.BlueROV_PIML_2 import BlueROV_PIML
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag
import matplotlib.pyplot as plt
import torch


class VEHICLE_SIM:

    def __init__(self, model_type: str, dt: float, init_pose: list, time_vec: list, control_vec: list, vehicle: str):

        # Initial conditions 
        self.x0 = init_pose

        # Time step
        self.dt = dt

        # Create vehicle instance
        if vehicle == "SAM":
            self.vehicle = SAM_PIML(dt=dt, piml_type=model_type)
        elif vehicle == "BROV":
            self.vehicle = BlueROV_PIML(h=dt, piml_type=model_type)
        else: 
            print("Selected vehicle for VEHICLE_SIM does not exist")
            return
        
        # Calculating how many sim steps we need to cover full time
        self.n_sim = int((int(max(time_vec)) - int(min(time_vec)))/self.dt)

        # Time tracking for when to update controls
        self.idx = 0
        self.times = time_vec
        self.t = time_vec[self.idx].item()
        self.controls = control_vec

    def run_sim(self):
        
        print(f" Running simulator...")
        # For storing results of sim
        data = np.empty((len(self.x0), self.n_sim))
        data[:,0] = self.x0
        idx_max = len(self.controls)

        # Euler forward integration
        for i in range(self.n_sim-1):
        
            data[:,i+1] = data[:,i] + self.vehicle.dynamics(data[:,i], self.controls[self.idx]) * (self.dt)
            self.t += self.dt
            # Update index for controls when we have new data based on current time
            if self.t > self.times[self.idx]:
                self.idx += 1
                self.idx = min([self.idx, idx_max-1]) # At the end of sim we just run with the last control input
        return data


if __name__ == "__main__":
    print(f" Initializing simulator...")
    
    # Ground truth data (SAM)
    eta_gt, nu_gt, u_fb_gt, u_cmd_gt, Dv_comp_gt, Mv_dot_gt, Cv_gt, g_eta_gt, tau_gt, t_gt = load_data_from_bag("src/smarc_modelling/piml/data/rosbags/rosbag_validate", "torch")
    init_pose = torch.Tensor.tolist(torch.cat([eta_gt[0], nu_gt[0], u_fb_gt[0]]))

    # Simulator parameters
    dt = np.mean(np.diff(t_gt)) # Time step 
    dt_model = 0.01
    
    # Models for simulation
    print(f" Initalizing models...")
    sam_wb = VEHICLE_SIM(None, dt_model, init_pose, t_gt, u_cmd_gt, "SAM")
    sam_pinn = VEHICLE_SIM("pinn", dt_model, init_pose, t_gt, u_cmd_gt, "SAM")

    print(f" Running simulator...")
    # Running simulations
    eta_wb = torch.Tensor(sam_wb.run_sim()[0:7, :]).T
    print(f" Done with white-box sim...")
    eta_pinn = torch.Tensor(sam_pinn.run_sim()[0:7, :]).T
    print(f" Done with PINN sim...")

    print(f" All sims done! Making plots.")

    # Plotting trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for vector, label in zip([eta_gt], ["Ground Truth"]):
        # Plotting trajectories
        points = vector[:, :3].T
        # Attaching 3D axis to the figure
        ax.plot(points[0], points[1], points[2], label=label)

    # Settings for plot
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.legend()
    plt.show()