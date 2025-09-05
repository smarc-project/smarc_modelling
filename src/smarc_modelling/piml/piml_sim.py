#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from smarc_modelling.vehicles.SAM_PIML import SAM_PIML
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_deg
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

        # Decide if we are going to update the state or not
        self.state_update = state_update

    def run_sim(self):
        print(f" Running simulator...")
        
        # For storing results of sim
        data = np.empty((len(self.x0), self.n_sim))
        data[:, 0] = self.x0

        # For getting last value before quat errors
        end_val = len(self.controls)
        once = True
        time_since_update = 0
        times =[]
        
        for i in range(self.n_sim-1):

            if i % 3 == 0 and self.state_update:
                data[:, i] = torch.Tensor.tolist(torch.cat([states[0][i], states[1][i], states[2][i]]))
                times.append(time_since_update)
                time_since_update = 0

            # Get the current time step
            dt = self.var_dt[i]
            self.vehicle.update_dt(dt)
            time_since_update += dt

            # Do sim step using ef
            try:
                data[:, i+1] = self.ef(data[:, i], self.controls[i], dt, self.vehicle.dynamics)
            except:
                data[:, i+1] = data[:, i-10]
                if once:
                    once = False
                    end_val = i - 100

        if self.state_update:
            print(f" Average times between resets: {np.mean(times)}")

        return data, end_val
    
    def rk4(self, x, u, dt, fun):
        # https://github.com/smarc-project/smarc_modelling/blob/master/src/smarc_modelling/sam_sim.py#L38C1-L46C15 
        # Runge Kutta 4
        k1 = fun(x, u)
        k2 = fun(x+dt/2*k1, u)
        k3 = fun(x+dt/2*k2, u)
        k4 = fun(x+dt*k3, u)
        dxdt = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return dxdt

    def ef(self, x, u, dt, fun):
        dxdt = x + fun(x, u) * dt
        return dxdt
    
if __name__ == "__main__":
    print(f" Starting simulator...")

    # Loading ground truth data
    eta, nu, u_fb, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, nu_dot = load_data_from_bag("src/smarc_modelling/piml/data/rosbags/rosbag_9", "torch")
    print(eta.shape)
    eta = eta[75:, :]
    nu = nu[75:, :]
    u_fb = u_fb[75:, :]
    states = [eta, nu, u_fb]

    # Initial positions for flipping frames
    y0 = eta[0, 1].item()
    z0 = eta[0, 2].item()
 
    # Setting up model for simulations
    reset_state = False
    sam_wb = SIM(None, states, t, u_cmd, reset_state) # White-box
    sam_pinn = SIM("pinn", states, t, u_cmd, reset_state) # Physics Informed Neural Network 
    sam_nn = SIM("nn", states, t, u_cmd, reset_state) # Standard Neural Network
    sam_naive_nn = SIM("naive_nn", states, t, u_cmd, reset_state) # Naive NN
    sam_bpinn = SIM("bpinn", states, t, u_cmd, reset_state) # Bayesian Physics Informed Neural Network
    
    # Running the simulators
    print(f" Running white-box simulation...")
    start_time = time.time()
    results_wb, end_val_wb = sam_wb.run_sim()
    end_time = time.time()
    results_wb = torch.tensor(results_wb).T
    eta_wb = results_wb[:, 0:7]
    nu_wb = results_wb[:, 7:13]
    print(f" White-box inference time: {(end_time-start_time)*1000/end_val_wb}")
    print(f" Done with the white-box sim!")

    # print(f" Running PINN simulation...")
    # start_time = time.time()
    # results_pinn, end_val_pinn = sam_pinn.run_sim()
    # end_time = time.time()
    # results_pinn = torch.tensor(results_pinn).T
    # eta_pinn = results_pinn[:, 0:7]
    # nu_pinn = results_pinn[:, 7:13]
    # print(f" PINN inference time: {(end_time-start_time)*1000/end_val_pinn}")
    # print(f" Done with the PINN sim!")

    # print(f" Running NN simulation...")
    # start_time = time.time()
    # results_nn, end_val_nn = sam_nn.run_sim()
    # end_time = time.time()
    # results_nn = torch.tensor(results_nn).T
    # eta_nn = results_nn[:, 0:7]
    # nu_nn = results_nn[:, 7:13]
    # print(f" NN inference time: {(end_time-start_time)*1000/end_val_nn}")
    # print(f" Done with the NN sim!")

    # print(f" Running naive NN simulation...")
    # # start_time = time.time()
    # results_naive_nn, end_val_naive_nn = sam_naive_nn.run_sim()
    # end_time = time.time()
    # results_naive_nn = torch.tensor(results_naive_nn).T
    # eta_naive_nn = results_naive_nn[:, 0:7]
    # nu_naive_nn = results_naive_nn[:, 7:13]
    # print(f" Naive NN inference time: {(end_time-start_time)*1000/end_val_naive_nn}")
    # print(f" Done with the naive NN sim!")

    # print(f" Running B-PINN simulation...")
    # start_time = time.time()
    # results_bpinn, end_val_bpinn = sam_bpinn.run_sim()
    # end_time = time.time()
    # results_bpinn = torch.tensor(results_bpinn).T
    # eta_bpinn = results_bpinn[:, 0:7]
    # nu_bpinn = results_bpinn[:, 7:13]
    # print(f" B-PINN inference time: {(end_time-start_time)*1000/end_val_bpinn}")
    # print(f" Done with the B-PINN sim!")

    print(f" Done with all sims making plots!")

    # # Making real life down be down
    # eta[:, 2] = 2 * z0 - eta[:, 2]
    # eta_wb[:, 2] = 2 * z0 - eta_wb[:, 2]
    # eta_pinn[:, 2] = 2 * z0 - eta_pinn[:, 2]
    # eta_bpinn[:, 2] = 2 * z0 - eta_bpinn[:, 2]
    # eta_nn[:, 2] = 2 * z0 - eta_nn[:, 2]
    # eta_naive_nn[:, 2] = 2 * z0 - eta_naive_nn[:, 2]

    end_val = int(np.min([end_val_wb]))#, end_val_pinn, end_val_nn, end_val_naive_nn]))
    print(end_val)
    plt.style.use('science')

    # 3D trajectory plot
    if True:
        # Plotting trajectory in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # states = [eta, eta_wb, eta_pinn, eta_nn, eta_naive_nn, eta_bpinn]
        # state_names =  ["Ground Truth", "White-Box", "PINN", "NN", "Naive NN", "B-PINN"]

        states = [eta, eta_wb]
        state_names =  ["Ground Truth", "White-Box"]

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
        # Quat to deg
        eta_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta[:end_val]])
        eta_wb_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_wb[:end_val]])
        eta_pinn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_pinn[:end_val]])
        eta_nn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_nn[:end_val]])
        eta_naive_nn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_naive_nn[:end_val]])
        eta_bpinn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_bpinn[:end_val]])

        # Errors WB
        eta_wb_mse = (eta_wb_deg - eta_deg)**2
        nu_wb_mse = (nu_wb - nu)**2

        # Errors PINN
        eta_pinn_mse = (eta_pinn_deg - eta_deg)**2
        nu_pinn_mse = (nu_pinn - nu)**2

        # Errors NN
        eta_nn_mse = (eta_nn_deg - eta_deg)**2
        nu_nn_mse = (nu_nn - nu)**2

        # Errors naive NN
        eta_naive_nn_mse = (eta_naive_nn_deg - eta_deg)**2
        nu_naive_nn_mse = (nu_naive_nn - nu)**2

        # Errors B-PINN
        eta_bpinn_mse = (eta_bpinn_deg - eta_deg)**2
        nu_bpinn_mse = (nu_bpinn - nu)**2

        # Cumulative error
        eta_wb_error = np.cumsum(eta_wb_mse, axis=0)
        nu_wb_error = np.cumsum(nu_wb_mse, axis=0)
        eta_pinn_error = np.cumsum(eta_pinn_mse, axis=0)
        nu_pinn_error = np.cumsum(nu_pinn_mse, axis=0)
        eta_nn_error = np.cumsum(eta_nn_mse, axis=0)
        nu_nn_error = np.cumsum(nu_nn_mse, axis=0)
        eta_naive_nn_error = np.cumsum(eta_naive_nn_mse, axis=0)
        nu_naive_nn_error = np.cumsum(nu_naive_nn_mse, axis=0)
        eta_bpinn_error = np.cumsum(eta_bpinn_mse, axis=0)
        nu_bpinn_error = np.cumsum(nu_bpinn_mse, axis=0)

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "yaw", "pitch", "roll"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]
        labels_error = ["Cumulative Squared Error [$m^2$]", "Cumulative Squared Error [$m^2$]", "Cumulative Squared Error [$m^2$]", 
                        "Cumulative Squared Error [$\circ^2$]", "Cumulative Squared Error [$\circ^2$]", "Cumulative Squared Error [$\circ^2$]", 
                        "Cumulative Squared Error [$(m/s)^2$]", "Cumulative Squared Error [$(m/s)^2$]", "Cumulative Squared Error [$(m/s)^2$]",
                        "Cumulative Squared Error [$(\circ/s)^2$]", "Cumulative Squared Error [$(\circ/s)^2$]", "Cumulative Squared Error [$(\circ/s)^2$]"]

        # Plotting error in eta
        for i in range(6):
            # axes[i].set_yscale('log')
            axes[i].plot(eta_wb_error[:end_val, i], label="White-box")
            axes[i].plot(eta_pinn_error[:end_val, i], label="PINN", linestyle=":")
            # axes[i].plot(eta_bpinn_error[:end_val, i], label="B-PINN", linestyle=(0, (1, 1)))
            # axes[i].plot(eta_nn_error[:end_val, i], label="NN", linestyle="--")
            # axes[i].plot(eta_naive_nn_error[:end_val, i], label="Naive NN", linestyle="-.")
            axes[i].set_title(f"{labels_eta[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_error[i])
            axes[i].legend()

        plt.tight_layout()

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot nu errors
        for j in range(6):
            # axes[j+6].set_yscale('log')
            axes[j].plot(nu_wb_error[:end_val, j], label="White-box")
            axes[j].plot(nu_pinn_error[:end_val, j], label="PINN", linestyle=":")
            # axes[j].plot(nu_bpinn_error[:end_val, i], label="B-PINN", linestyle=(0, (1, 1)))
            # axes[j].plot(nu_nn_error[:end_val, i], label="NN", linestyle="--")
            # axes[j].plot(nu_naive_nn_error[:end_val, i], label="Naive NN", linestyle="-.")
            axes[j].set_title(f"{labels_nu[j]}")
            axes[j].set_xlabel("Timestep")
            axes[j].set_ylabel(labels_error[j+6])
            axes[j].legend()

        plt.tight_layout()

        print(f" PINN: {-100*(1 - np.concatenate( (eta_pinn_error[-1, :]/eta_wb_error[-1, :], nu_pinn_error[-1, :]/nu_wb_error[-1, :])))}")
        print(f" BPINN: {-100*(1 - np.concatenate( (eta_bpinn_error[-1, :]/eta_wb_error[-1, :], nu_bpinn_error[-1, :]/nu_wb_error[-1, :])))}")
        print(f" NN: {-100*(1 - np.concatenate( (eta_nn_error[-1, :]/eta_wb_error[-1, :], nu_nn_error[-1, :]/nu_wb_error[-1, :])))}")
        print(f" Naive NN: {-100*(1 - np.concatenate( (eta_naive_nn_error[-1, :]/eta_wb_error[-1, :], nu_naive_nn_error[-1, :]/nu_wb_error[-1, :])))}")

    # Plots of each state
    if True:
        # Quat to deg
        eta_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta[:end_val]])
        eta_wb_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_wb[:end_val]])
        # eta_pinn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_pinn[:end_val]])
        # eta_nn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_nn[:end_val]])
        # eta_naive_nn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_naive_nn[:end_val]])
        # eta_bpinn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_bpinn[:end_val]])

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "yaw", "pitch", "roll"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]
        labels_unit = ["State [m]", "State [m]", "State [m]",
                       "State [$\circ$]", "State [$\circ$]", "State [$\circ$]",
                       "State [m/s]", "State [m/s]", "State [m/s]",
                       "State [$\circ/s$]", "State [$\circ/s$]", "State [$\circ/s$]"]

        # Plotting error in eta
        for i in range(6):
            axes[i].plot(eta_deg[:end_val, i], label="Ground Truth")
            axes[i].plot(eta_wb_deg[:end_val, i], label="White-box")
            # axes[i].plot(eta_pinn_deg[:end_val, i], label="PINN")
            # axes[i].plot(eta_bpinn_deg[:end_val, i], label="B-PINN")
            # axes[i].plot(eta_nn_deg[:end_val, i], label="NN")
            # axes[i].plot(eta_naive_nn_deg[:end_val, i], label="Naive NN")
            axes[i].set_title(f"{labels_eta[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_unit[i])
            axes[i].legend()

        plt.tight_layout()


        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot nu errors
        for j in range(6):
            axes[j].plot(nu[:end_val, j], label="Ground Truth")
            axes[j].plot(nu_wb[:end_val, j], label="White-box")
            # axes[j].plot(nu_pinn[:end_val, j], label="PINN")
            # axes[j].plot(nu_bpinn[:end_val, j], label="B-PINN")
            # axes[j].plot(nu_nn[:end_val, j], label="NN")
            # axes[j].plot(nu_naive_nn[:end_val, j], label="Naive NN")
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