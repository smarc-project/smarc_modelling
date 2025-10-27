#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from smarc_modelling.vehicles.SAM_PIML import SAM_PIML
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_rad, angle_diff
import matplotlib as mpl
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
        self.data = np.empty((len(self.x0), self.n_sim))
        self.vels = np.empty((6, self.n_sim))
        self.states = states

    def run_sim(self):
        print(f" Running simulator...")
        
        # For storing results of sim
        self.data[:, 0] = self.x0

        # For getting last value before quat errors
        end_val = len(self.controls)
        once = True
        time_since_update = 0
        times =[]
        
        for i in range(self.n_sim-1):
            
            # For resetting the state every n step, currently unused
            if i % 3 == 0 and self.state_update:
                self.data[:, i] = torch.Tensor.tolist(torch.cat([self.states[0][i], self.states[1][i], self.states[2][i]]))
                times.append(time_since_update)
                time_since_update = 0

            # Get the current time step
            dt = self.var_dt[i]
            self.vehicle.update_dt(dt)
            time_since_update += dt

            if False: # Direct velocity prediction, set to if False unless explicitly needed
                x = torch.Tensor.tolist(torch.cat([self.states[0][i], self.states[1][i], self.states[2][i]]))
                u = self.controls[i]
                self.vels[:, i] = self.vehicle.dynamics(x, u)[7:13]

            # Do sim step using ef
            try:
                self.data[:, i+1] = self.rk4(self.data[:, i], self.controls[i], dt, self.vehicle.dynamics)
            except:
                self.data[:, i+1] = self.data[:, i]
                if once:
                    once = False
                    end_val = i - 5

        if self.state_update:
            print(f" Average times between resets: {np.mean(times)}")

        return self.data, end_val, self.vels
    
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
    eta, nu, u_fb, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, nu_dot = load_data_from_bag("src/smarc_modelling/piml/data/rosbags/evaluate_1", "torch")
    
    start_val = 0
    eta = eta[start_val:, :]
    nu = nu[start_val:, :]
    u_fb = u_fb[start_val:, :]
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
    
    # Running the simulators
    # WB sim and organization of results
    print(f" Running white-box simulation...")
    start_time = time.time()
    results_wb, end_val_wb, vels_wb = sam_wb.run_sim()
    end_time = time.time()
    results_wb = torch.tensor(results_wb).T
    eta_wb = results_wb[:, 0:7]
    nu_wb = results_wb[:, 7:13]
    print(f" White-box inference time: {(end_time-start_time)*1000/end_val_wb}")
    print(f" Done with the white-box sim!")

    # PINN sim and organization of results
    print(f" Running PINN simulation...")
    start_time = time.time()
    results_pinn, end_val_pinn, vels_pinn = sam_pinn.run_sim()
    end_time = time.time()
    results_pinn = torch.tensor(results_pinn).T
    eta_pinn = results_pinn[:, 0:7]
    nu_pinn = results_pinn[:, 7:13]
    print(f" PINN inference time: {(end_time-start_time)*1000/end_val_pinn}")
    print(f" Done with the PINN sim!")

    # NN sim and organization of results
    print(f" Running NN simulation...")
    start_time = time.time()
    results_nn, end_val_nn, vels_nn = sam_nn.run_sim()
    end_time = time.time()
    results_nn = torch.tensor(results_nn).T
    eta_nn = results_nn[:, 0:7]
    nu_nn = results_nn[:, 7:13]
    print(f" NN inference time: {(end_time-start_time)*1000/end_val_nn}")
    print(f" Done with the NN sim!")

    # Naive NN sim and organization of results
    print(f" Running naive NN simulation...")
    start_time = time.time()
    results_naive_nn, end_val_naive_nn, vels_naive_nn = sam_naive_nn.run_sim()
    end_time = time.time()
    results_naive_nn = torch.tensor(results_naive_nn).T
    eta_naive_nn = results_naive_nn[:, 0:7]
    nu_naive_nn = results_naive_nn[:, 7:13]
    print(f" Naive NN inference time: {(end_time-start_time)*1000/end_val_naive_nn}")
    print(f" Done with the naive NN sim!")

    print(f" Done with all sims making plots!")

    end_val = int(np.min([end_val_wb, end_val_pinn, end_val_nn, end_val_naive_nn]))
    print(end_val)
    plt.style.use('science')

    # 3D trajectory plot
    if False:
        # Plotting trajectory in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
     
        states = [eta, eta_wb, eta_pinn, eta_nn, eta_naive_nn]
        state_names =  ["Ground Truth", "White-Box", "PINN", "NN", "Naive NN"]

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
        eta_wb_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_wb[:end_val]])
        eta_pinn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_pinn[:end_val]])
        eta_nn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_nn[:end_val]])
        eta_naive_nn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_naive_nn[:end_val]])

        # Errors WB
        eta_wb_mse_pos = (eta_wb_rad[:, 0:3] - eta_rad[:, 0:3])**2
        eta_wb_mse_ang = angle_diff(eta_wb_rad[:, 3:6], eta_rad[:, 3:6])**2
        eta_wb_mse = np.hstack([eta_wb_mse_pos, eta_wb_mse_ang])
        nu_wb_mse = (nu_wb - nu)**2

        # Errors PINN
        eta_pinn_mse_pos = (eta_pinn_rad[:, 0:3] - eta_rad[:, 0:3])**2
        eta_pinn_mse_ang = angle_diff(eta_pinn_rad[:, 3:6], eta_rad[:, 3:6])**2
        eta_pinn_mse = np.hstack([eta_pinn_mse_pos, eta_pinn_mse_ang])
        nu_pinn_mse = (nu_pinn - nu)**2

        # Errors NN
        eta_nn_mse_pos = (eta_nn_rad[:, 0:3] - eta_rad[:, 0:3])**2
        eta_nn_mse_ang = angle_diff(eta_nn_rad[:, 3:6], eta_rad[:, 3:6])**2
        eta_nn_mse = np.hstack([eta_nn_mse_pos, eta_nn_mse_ang])
        nu_nn_mse = (nu_nn - nu)**2

        # Errors naive NN
        eta_naive_nn_mse_pos = (eta_naive_nn_rad[:, 0:3] - eta_rad[:, 0:3])**2
        eta_naive_nn_mse_ang = angle_diff(eta_naive_nn_rad[:, 3:6], eta_rad[:, 3:6])**2
        eta_naive_nn_mse = np.hstack([eta_naive_nn_mse_pos, eta_naive_nn_mse_ang])
        nu_naive_nn_mse = (nu_naive_nn - nu)**2

        # Cumulative error
        eta_wb_error = np.cumsum(eta_wb_mse, axis=0) * 1/end_val
        nu_wb_error = np.cumsum(nu_wb_mse, axis=0) * 1/end_val
        eta_pinn_error = np.cumsum(eta_pinn_mse, axis=0) * 1/end_val
        nu_pinn_error = np.cumsum(nu_pinn_mse, axis=0) * 1/end_val
        eta_nn_error = np.cumsum(eta_nn_mse, axis=0) * 1/end_val
        nu_nn_error = np.cumsum(nu_nn_mse, axis=0) * 1/end_val
        eta_naive_nn_error = np.cumsum(eta_naive_nn_mse, axis=0) * 1/end_val
        nu_naive_nn_error = np.cumsum(nu_naive_nn_mse, axis=0) * 1/end_val

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "roll", "pitch", "yaw"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]
        labels_error = ["Mean Squared Error [$m^2$]", "Mean Squared Error [$m^2$]", "Mean Squared Error [$m^2$]", 
                        "Mean Squared Error [$rad^2$]", "Mean Squared Error [$rad^2$]", "Mean Squared Error [$rad^2$]", 
                        "Mean Squared Error [$(m/s)^2$]", "Mean Squared Error [$(m/s)^2$]", "Mean Squared Error [$(m/s)^2$]",
                        "Mean Squared Error [$(rad/s)^2$]", "Mean Squared Error [$(rad/s)^2$]", "Mean Squared Error [$(rad/s)^2$]"]

        # Plotting error in eta
        for i in range(6):
            # axes[i].set_yscale('log')
            axes[i].plot(eta_wb_error[:end_val, i], label="White-box")
            axes[i].plot(eta_pinn_error[:end_val, i], label="PINN", linestyle=":")
            axes[i].plot(eta_nn_error[:end_val, i], label="NN", linestyle="--")
            axes[i].plot(eta_naive_nn_error[:end_val, i], label="Naive NN", linestyle="-.")
            axes[i].set_title(f"{labels_eta[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_error[i])
            axes[i].legend()

        plt.tight_layout()

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot nu errors
        for i in range(6):
            axes[i].plot(nu_wb_error[:end_val, i], label="White-box")
            axes[i].plot(nu_pinn_error[:end_val, i], label="PINN", linestyle=":")
            axes[i].plot(nu_nn_error[:end_val, i], label="NN", linestyle="--")
            axes[i].plot(nu_naive_nn_error[:end_val, i], label="Naive NN", linestyle="-.")
            axes[i].set_title(f"{labels_nu[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_error[i+6])
            axes[i].legend()

        plt.tight_layout()

        end_val -= 1
        print(f" White-box: {nu_wb_error[end_val, :]}")
        print(f" PINN: {nu_pinn_error[end_val, :]}")
        print(f" NN: {nu_nn_error[end_val, :]}")
        print(f" Naive NN: {nu_naive_nn_error[end_val, :]}")

    # Plots of each state
    if False:
        # Quat to deg
        eta_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta[:end_val]])
        eta_wb_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_wb[:end_val]])
        eta_pinn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_pinn[:end_val]])
        eta_nn_deg = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_nn[:end_val]])
        eta_naive_nn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_naive_nn[:end_val]])

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "roll", "pitch", "yaw"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]
        labels_unit = ["State [m]", "State [m]", "State [m]",
                       "State [$rad$]", "State [$rad$]", "State [$rad$]",
                       "State [m/s]", "State [m/s]", "State [m/s]",
                       "State [$rad/s$]", "State [$rad/s$]", "State [$rad/s$]"]
      
        # Plotting eta pos
        for i in range(6):
            axes[i].plot(eta_rad[:end_val, i], label="Ground Truth", color="#aa3377")
            axes[i].plot(eta_wb_rad[:end_val, i], label="White-box")
            axes[i].plot(eta_pinn_rad[:end_val, i], label="PINN", linestyle=":")
            axes[i].plot(eta_nn_deg[:end_val, i], label="NN", linestyle="--")
            axes[i].plot(eta_naive_nn_rad[:end_val, i], label="Naive NN", linestyle="-.")
            axes[i].set_title(f"{labels_eta[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_unit[i])
            axes[i].legend()

        plt.tight_layout()


        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot nu value
        for i in range(6):
            axes[i].plot(nu[:end_val, i], label="Ground Truth", color="#aa3377")
            axes[i].plot(nu_wb[:end_val, i], label="White-box")
            axes[i].plot(nu_pinn[:end_val, i], label="PINN", linestyle=":")
            axes[i].plot(nu_nn[:end_val, i], label="NN", linestyle="--")
            axes[i].plot(nu_naive_nn[:end_val, i], label="Naive NN", linestyle="-.")
            axes[i].set_title(f"{labels_nu[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel(labels_unit[i+6])
            axes[i].legend()

        plt.tight_layout()

    # Compute final RMSE for x,y,z pos
    if False:
        from sklearn import metrics as met
     
        N = end_val # Amount of data points we simulated
        
        # State in radians for angles
        eta_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta[:end_val]])
        eta_wb_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_wb[:end_val]])
        eta_pinn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_pinn[:end_val]])
        eta_nn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_nn[:end_val]])
        eta_naive_nn_rad = np.array([eta_quat_to_rad(eta_vec) for eta_vec in eta_naive_nn[:end_val]])

        # WB RMSE
        wb_rmse = met.root_mean_squared_error(eta[:end_val, 0:3], eta_wb[:end_val, 0:3])
        ang_diff = angle_diff(eta_wb_rad[:, 3:6], eta_rad[:, 3:6])**2
        wb_rmse_ang = met.root_mean_squared_error(np.zeros_like(ang_diff), ang_diff)
        print(f"WB RMSE - position: {wb_rmse} - angle: {wb_rmse_ang}")

        # PINN RMSE
        pinn_rmse = met.root_mean_squared_error(eta[:end_val, 0:3], eta_pinn[:end_val, 0:3])
        ang_diff = angle_diff(eta_pinn_rad[:, 3:6], eta_rad[:, 3:6])**2
        pinn_rmse_ang = met.root_mean_squared_error(np.zeros_like(ang_diff), ang_diff)
        print(f"PINN RMSE - position: {pinn_rmse} - angle: {pinn_rmse_ang}")
        
        # NN RMSE
        nn_rmse = met.root_mean_squared_error(eta[:end_val, 0:3], eta_nn[:end_val, 0:3])
        ang_diff = angle_diff(eta_nn_rad[:, 3:6], eta_rad[:, 3:6])**2
        nn_rmse_ang = met.root_mean_squared_error(np.zeros_like(ang_diff), ang_diff)
        print(f"NN RMSE - position: {nn_rmse} - angle: {nn_rmse_ang}")

        # Naive NN RMSE
        naive_nn_rmse = met.root_mean_squared_error(eta[:end_val, 0:3], eta_naive_nn[:end_val, 0:3])
        ang_diff = angle_diff(eta_naive_nn_rad[:, 3:6], eta_rad[:, 3:6])**2
        naive_nn_rmse_ang = met.root_mean_squared_error(np.zeros_like(ang_diff), ang_diff)
        print(f"Naive NN RMSE - position: {naive_nn_rmse} - angle: {naive_nn_rmse_ang}")

    # MSE bar plots for direct acceleration prediction
    if False:
        nu_dot = np.array(nu_dot)

        # MSE
        mse_wb = np.mean((nu_dot[:end_val, :] - vels_wb[:, :end_val].T)**2, axis=0)
        mse_pinn = np.mean((nu_dot[:end_val, :] - vels_pinn[:, :end_val].T)**2, axis=0)
        mse_nn = np.mean((nu_dot[:end_val, :] - vels_nn[:, :end_val].T)**2, axis=0)
        mse_naive_nn = np.mean((nu_dot[:end_val, :] - vels_naive_nn[:, :end_val].T)**2, axis=0)

        # Formatting for plot
        mse_all = np.stack([mse_wb, mse_pinn, mse_nn, mse_naive_nn], axis=1)
        models = ["White-box", "PINN", "NN", "Naive NN"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]
        labels_unit = ["MSE [m/s]", "MSE [m/s]", "MSE [m/s]", "MSE [rad/s]", "MSE [rad/s]", "MSE [rad/s]"]
        y_max = [0.5, 0.003, 0.004, 25, 3, 0.08]
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.bar(models, mse_all[i, :], color=["blue", "green", "orange", "red"])
            ax.set_title(f"{labels_nu[i]}")
            ax.set_ylabel(f"{labels_unit[i]}")
            ax.set_ylim(0, y_max[i])
            ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()

plt.show()