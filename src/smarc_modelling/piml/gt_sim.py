#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from smarc_modelling.vehicles.SAM_PIML import SAM_PIML
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_rad, angular_vel_to_quat_vel, eta_quat_to_deg
from smarc_modelling.lib.gnc import Rzyx, Tzyx
import matplotlib.pyplot as plt
import torch
import scienceplots # For fancy plotting
import time

class SIM:
    """Simulator for SAM / other UAVs"""

    def __init__(self, piml_type: str, states: list, time_vec: list, control_vec: list, state_update: bool):

        # Initial pose
        self.x0 = torch.Tensor.tolist(torch.cat([states[0][0], states[1][0], states[2][0]]))

        # Create vehicle instance
        self.vehicle = SAM_PIML(dt=0.01, piml_type=piml_type)

        # Controls and sim variables
        self.controls = control_vec
        self.n_sim = np.shape(time_vec)[0]
        self.var_dt = np.diff(time_vec)
        print(self.var_dt[45:60])

        # Decide if we are going to update the state or not
        self.state_update = state_update
        self.data = np.empty((len(self.x0), self.n_sim))
        self.states = states

    def run_sim(self):
        print(f" Running simulator...")
        
        # For storing results of sim
        self.data[:, 0] = self.x0

        # For getting last value before quat errors
        end_val = len(self.controls)
        time_since_update = 0
        times = []
        
        for i in range(self.n_sim-1):

            dt = self.var_dt[i]
        
            if i % 3 == 0 and self.state_update:
                self.data[:, i] = torch.Tensor.tolist(torch.cat([self.states[0][i], self.states[1][i], self.states[2][i]]))
                times.append(time_since_update)
                time_since_update = 0

   
            eta_sim = self.data[0:7, i]
            nu_sim = self.data[7:13, i]

            # Getting "predictions"
            nu_dot = self.states[3][i] # GT acceleration        
            eta_dot_body = self.states[1][i] # GT speed
            eta = self.states[0][i] # GT pose

            eta_ang = eta_quat_to_rad(eta)

            # Convert to global frame
            [x, y, z] = np.matmul(Rzyx(eta_ang[3], eta_ang[4], eta_ang[5]), eta_dot_body[0:3])
            [roll, pitch, yaw] = np.matmul(Tzyx(eta_ang[3], eta_ang[4]), eta_dot_body[0:3])
            eta_dot = np.hstack([x, y, z, -roll, -pitch, yaw])
            eta_dot = angular_vel_to_quat_vel(eta_ang, eta_dot)


            x_dot = np.concatenate([eta_dot, nu_dot, [0, 0, 0, 0, 0, 0]]) # Controls just 0 here since we dont use them
            # --------------------------------- #

            # EF
            self.data[:, i+1] =  self.data[:, i] + x_dot * dt

        if self.state_update:
            print(f" Average times between resets: {np.mean(times)}")

        return self.data, end_val

if __name__ == "__main__":
    print(f" Starting simulator...")

    # Loading ground truth data
    eta, nu, u_fb, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, nu_dot = load_data_from_bag("src/smarc_modelling/piml/data/rosbags/rosbag_9", "torch")
    print(nu_dot.shape)
    start_val = 5
    eta = eta[start_val:, :]
    nu = nu[start_val:, :]
    u_fb = u_fb[start_val:, :]
    nu_dot = nu_dot[start_val:, :]
    t = t[start_val:]
    states = [eta, nu, u_fb, nu_dot]

    # Setting up model for simulations
    reset_state = False
    sim = SIM(None, states, t, u_cmd, reset_state)

    # Running the simulators
    print(f" Running white-box simulation...")
    start_time = time.time()
    results_gt, end_val_gt = sim.run_sim()
    end_time = time.time()
    results_gt = torch.tensor(results_gt).T
    eta_gt = results_gt[:, 0:7]
    nu_gt = results_gt[:, 7:13]
    print(f" White-box inference time: {(end_time-start_time)*1000/end_val_gt}")
    print(f" Done with the white-box sim!")

    print(f" Done with all sims making plots!")

    end_val = end_val_gt
    print(end_val)
    plt.style.use('science')
    end_val = 60

    # 3D trajectory plot
    if True:
        # Plotting trajectory in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # states = [eta, eta_wb, eta_pinn, eta_nn, eta_naive_nn, eta_bpinn]
        # state_names =  ["Ground Truth", "White-Box", "PINN", "NN", "Naive NN", "B-PINN"]

        states = [eta, eta_gt]
        state_names =  ["Ground Truth", "GT model"]

        for vector, label in zip(states, state_names):
            # Plotting trajectory
            vector = np.array(vector)
            points = vector[:end_val, :3].T
            ax.plot(points[0], points[1], points[2], label=label)

        # Equal axis scaling
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        max_range = max(x_range, y_range, z_range)

        # Midpoints
        x_middle = 0.5 * (xlim[0] + xlim[1])
        y_middle = 0.5 * (ylim[0] + ylim[1])
        z_middle = 0.5 * (zlim[0] + zlim[1])

        # Set new limits
        ax.set_xlim(x_middle - max_range/2, x_middle + max_range/2)
        ax.set_ylim(y_middle - max_range/2, y_middle + max_range/2)
        ax.set_zlim(z_middle - max_range/2, z_middle + max_range/2)

        # Axis labels
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        plt.legend()

    # Cumulative error plots
    if False:
        # Quat to rad
        eta_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta[:end_val]])
        eta_gt_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_gt[:end_val]])

        # Errors WB
        eta_gt_mse_pos = (eta_gt_rad[0:3] - eta_rad[0:3])**2
        eta_gt_mse_angs = ((eta_gt_rad[3:6] - eta_rad[3:6]) % 2*np.pi)**2
        eta_gt_mse = np.hstack([eta_gt_mse_pos, eta_gt_mse_angs])
        nu_gt_mse = (nu_gt - nu)**2

        # Cumulative error
        eta_gt_error = np.cumsum(eta_gt_mse, axis=0)
        nu_gt_error = np.cumsum(nu_gt_mse, axis=0)

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "yaw", "pitch", "roll"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]
        labels_error = ["Cumulative Squared Error [$m^2$]", "Cumulative Squared Error [$m^2$]", "Cumulative Squared Error [$m^2$]", 
                        "Cumulative Squared Error [$rad^2$]", "Cumulative Squared Error [$rad^2$]", "Cumulative Squared Error [$rad^2$]", 
                        "Cumulative Squared Error [$(m/s)^2$]", "Cumulative Squared Error [$(m/s)^2$]", "Cumulative Squared Error [$(m/s)^2$]",
                        "Cumulative Squared Error [$(rad/s)^2$]", "Cumulative Squared Error [$(rad/s)^2$]", "Cumulative Squared Error [$(rad/s)^2$]"]

        # Plotting error in eta
        for i in range(6):
            # axes[i].set_yscale('log')
            axes[i].plot(eta_gt_error[:end_val, i], label="White-box")
            axes[i].set_title(f"{labels_eta[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_error[i])
            axes[i].legend()

        plt.tight_layout()

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot nu errors
        for j in range(6):
            axes[j].plot(nu_gt_error[:end_val, j], label="White-box")
            axes[j].set_title(f"{labels_nu[j]}")
            axes[j].set_xlabel("Timestep")
            axes[j].set_ylabel(labels_error[j+6])
            axes[j].legend()

        plt.tight_layout()

    # Plots of each state
    if True:
        # Quat to deg
        eta_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta[:end_val]])
        eta_gt_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_gt[:end_val]])

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "yaw", "pitch", "roll"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]
        labels_unit = ["State [m]", "State [m]", "State [m]",
                       "State [$rad$]", "State [$rad$]", "State [$rad$]",
                       "State [m/s]", "State [m/s]", "State [m/s]",
                       "State [$rad/s$]", "State [$rad/s$]", "State [$rad/s$]"]

        # Plotting error in eta
        for i, j in enumerate([0, 1, 2, 5, 4, 3]):
            axes[i].plot(eta_rad[:end_val, j], label="Ground Truth")
            axes[i].plot(eta_gt_rad[:end_val, j], label="GT sim")
            axes[i].set_title(f"{labels_eta[j]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_unit[j])
            axes[i].legend()

        plt.tight_layout()


        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot nu errors
        for j in range(6):
            axes[j].plot(nu[:end_val, j], label="Ground Truth")
            axes[j].plot(nu_gt[:end_val, j], label="GT sim")
            axes[j].set_title(f"{labels_nu[j]}")
            axes[j].set_xlabel("Timestep")
            axes[j].set_ylabel(labels_unit[j+6])
            axes[j].legend()

        plt.tight_layout()

    # Displaying plots
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close("all") # <-- This does not work :/ gotta close it the manual way :(