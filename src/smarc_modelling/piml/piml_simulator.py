#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 9x5x3 qualisys arqus underwater
# Runs euler forward integration to simulate vehicles
# Currently works with
# White-Box models
# PINN models

import numpy as np
from smarc_modelling.vehicles.SAM_PIML import SAM_PIML # Customized SAM using PIML predictions for D
from smarc_modelling.vehicles.BlueROV_PIML_2 import BlueROV_PIML
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_deg
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
        Dv_vec = []
        x_acc_vec = []

        # Euler forward integration
        for i in range(self.n_sim-1):

            dxdt, Dv, x_acc = self.vehicle.dynamics(data[:,i], self.controls[self.idx])
            
            Dv_vec.append(Dv)
            x_acc_vec.append(x_acc)

            data[:,i+1] = data[:,i] + dxdt * (self.dt)
            self.t += self.dt
            # Update index for controls when we have new data based on current time
            if self.t > self.times[self.idx]:
                self.idx += 1
                self.idx = min([self.idx, idx_max-1]) # At the end of sim we just run with the last control input
        return data, Dv_vec, x_acc_vec


if __name__ == "__main__":
    print(f" Initializing simulator...")
    
    # Ground truth data (SAM)
    eta_gt, nu_gt, u_fb_gt, u_cmd_gt, Dv_comp_gt, Mv_dot_gt, Cv_gt, g_eta_gt, tau_gt, t_gt = load_data_from_bag("src/smarc_modelling/piml/data/rosbags/rosbag_tank_forward_blind", "torch")
    init_pose = torch.Tensor.tolist(torch.cat([eta_gt[0], nu_gt[0], u_fb_gt[0]]))
    
    # Fixing coordinates for gt data NOTE: Dont do this they should have the same frame as the sim!! Just flip everything upside down when doing visualization
    # zeroing = eta_gt[0, 0:3].clone()
    # eta_gt[:, 0:3] = eta_gt[:, 0:3] - zeroing # Move data origin to 0
    # eta_gt[:, 0] = -eta_gt[:, 0]
    # eta_gt[:, 1] = -eta_gt[:, 1]
    # eta_gt[:, 2] = -eta_gt[:, 2]
    # eta_gt[:, 0:3] = eta_gt[:, 0:3] + zeroing

    # Overwrite controls to check sim
    # "Neutral state"
    # vbs_neutral = 72.5 # With self.B = self.W + (self.vbs.m_vbs * 0.45 * self.g)
    # lcg_neutral = 50.62
    # u_cmd_gt[:, 0] = vbs_neutral # VBS
    # u_cmd_gt[:, 1] = lcg_neutral # LCG
    # u_cmd_gt[:, 2] = 0 # np.deg2rad(7)    # Vertical (stern)
    # u_cmd_gt[:, 3] = -np.deg2rad(7)   # Horizontal (rudder)
    # u_cmd_gt[:, 4] = 100 # 1000     # RPM 1
    # u_cmd_gt[:, 5] = 100 # 1000     # RPM 2
    # init_pose = np.zeros(19).squeeze()
    # init_pose[6] = 1.0
    # init_pose[13] = vbs_neutral
    # init_pose[14] = lcg_neutral

    # # "Spin"
    # u_cmd_gt[:, 0] = 50 # VBS
    # u_cmd_gt[:, 1] = 50 # LCG
    # u_cmd_gt[:, 2] = np.deg2rad(7)    # Vertical (stern)
    # u_cmd_gt[:, 3] = -np.deg2rad(7)   # Horizontal (rudder)
    # u_cmd_gt[:, 4] = 1000     # RPM 1
    # u_cmd_gt[:, 5] = 1000     # RPM 2

    # Simulator parameters
    dt = np.mean(np.diff(t_gt)) # Time step 
    print(f" Dataset average dt as: {dt}")
    dt_model = 0.01

    # Models for simulation
    print(f" Initalizing models...")
    sam_wb = VEHICLE_SIM(None, dt_model, init_pose, t_gt, u_cmd_gt, "SAM")
    sam_pinn = VEHICLE_SIM("pinn", dt_model, init_pose, t_gt, u_cmd_gt, "SAM")

    # Running simulations
    print(f" Running all the simulators...")

    # White-box
    results_wb, Dv_wb, x_acc_wb = sam_wb.run_sim()
    results_wb = torch.tensor(results_wb).T
    Dv_wb = np.array(Dv_wb)
    eta_wb = results_wb[:, 0:7]
    nu_wb = results_wb[:, 7:13]
    fb_wb = results_wb[:, 13:19]
    print(f" Done with white-box sim...")

    # PINN
    # eta_pinn = torch.Tensor(sam_pinn.run_sim()[0:7, :]).T
    # print(f" Done with PINN sim...")

    print(f" All sims done! Making plots.")
    
    # Picking out index corresponding to existing time data
    max_index_eta = np.shape(eta_wb)[0] - 1
    eta_times = torch.clamp((t_gt * (1/dt_model)).long(), min=0, max=max_index_eta)
    eta_wb_selected = [eta_wb[i, :] for i in eta_times]
    nu_wb_selected = [nu_wb[i, :] for i in eta_times]
    fb_wb_selected = [fb_wb[i, :] for i in eta_times]
    
    # Forces have one less value in them due to begin computed from a difference of velocities
    eta_times = torch.clamp(eta_times, min=0, max=max_index_eta-1)
    Dv_wb_selected = [Dv_wb[i, :] for i in eta_times]

    # Plotting trajectories in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for vector, label in zip([eta_gt, eta_wb], ["Ground Truth", "White-Box model"]):
        # Plotting trajectories
        vector = np.array(vector)
        points = vector[:, :3].T
        # Attaching 3D axis to the figure
        ax.plot(points[0], points[1], points[2], label=label)

    # Settings for plot
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.legend()

    
    eta_gt = [eta_quat_to_deg(eta_vec) for eta_vec in eta_gt] # Quat to angles for ground truth
    eta_wb_selected = [eta_quat_to_deg(eta_vec) for eta_vec in eta_wb_selected] # Quat to angles for synced data
    eta_wb = [eta_quat_to_deg(eta_vec) for eta_vec in eta_wb] # Quat to angles for unsynced data

    eta_gt = np.array(eta_gt)
    eta_wb_selected = np.array(eta_wb_selected)
    nu_wb_selected = np.array(nu_wb_selected)
    fb_wb_selected = np.array(fb_wb_selected)
    t_gt = np.array(t_gt)

    # Plotting each state
    if True:
        _, ax = plt.subplots(6, 3, figsize=(12, 10))

        # Pose
        ax[0,0].plot(t_gt, eta_gt[:, 0], label="x - gt")
        ax[0,0].plot(t_gt, eta_wb_selected[:, 0], label="x - wb")
        ax[0,0].legend()
        ax[0,1].plot(t_gt, eta_gt[:, 1], label="y - gt")
        ax[0,1].plot(t_gt, eta_wb_selected[:, 1], label="y - wb")
        ax[0,1].legend()
        ax[0,2].plot(t_gt, eta_gt[:, 2], label="z - gt")
        ax[0,2].plot(t_gt, eta_wb_selected[:, 2], label="z - wb")
        ax[0,2].legend()
        ax[1,0].plot(t_gt, eta_gt[:, 5], label="roll - gt")
        ax[1,0].plot(t_gt, eta_wb_selected[:, 5], label="roll - wb")
        ax[1,0].legend()
        ax[1,1].plot(t_gt, eta_gt[:, 4], label="pitch - gt")
        ax[1,1].plot(t_gt, eta_wb_selected[:, 4], label="pitch - wb")
        ax[1,1].legend()
        ax[1,2].plot(t_gt, eta_gt[:, 3], label="yaw - gt")
        ax[1,2].plot(t_gt, eta_wb_selected[:, 3], label="yaw - wb")
        ax[1,2].legend()
        ax[0,0].set_ylabel('x Position [m]')
        ax[0,1].set_ylabel('y Position [m]')
        ax[0,2].set_ylabel('-z Position [m]')
        ax[1,0].set_ylabel('roll [deg]')
        ax[1,1].set_ylabel('pitch [deg]')
        ax[1,2].set_ylabel('yaw [deg]')

        # Velocities
        ax[2,0].plot(t_gt, nu_gt[:, 0], label="u - gt")
        ax[2,0].plot(t_gt, nu_wb_selected[:, 0], label="u - wb")
        ax[2,0].legend()
        ax[2,1].plot(t_gt, nu_gt[:, 1], label="v - gt")
        ax[2,1].plot(t_gt, nu_wb_selected[:, 1], label="v - wb")
        ax[2,1].legend()
        ax[2,2].plot(t_gt, nu_gt[:, 2], label="w - gt")
        ax[2,2].plot(t_gt, nu_wb_selected[:, 2], label="w - wb")
        ax[2,2].legend()
        ax[3,0].plot(t_gt, nu_gt[:, 3], label="p - gt")
        ax[3,0].plot(t_gt, nu_wb_selected[:, 3], label="p - wb")
        ax[3,0].legend()
        ax[3,1].plot(t_gt, nu_gt[:, 4], label="q - gt")
        ax[3,1].plot(t_gt, nu_wb_selected[:, 4], label="q - wb")
        ax[3,1].legend()
        ax[3,2].plot(t_gt, nu_gt[:, 5], label="r - gt")
        ax[3,2].plot(t_gt, nu_wb_selected[:, 5], label="r - wb")
        ax[3,2].legend()
        ax[2,0].set_ylabel('u (x_dot)')
        ax[2,1].set_ylabel('v (y_dot)')
        ax[2,2].set_ylabel('w (z_dot)')
        ax[3,0].set_ylabel('p (roll_dot)')
        ax[3,1].set_ylabel('q (pitch_dot)')
        ax[3,2].set_ylabel('r (yaw_dot)')

        # Controls
        ax[4,0].plot(t_gt, u_fb_gt[:, 0], label="vbs - gt")
        ax[4,0].plot(t_gt, fb_wb_selected[:, 0], label="vbs - wb")
        ax[4,0].legend()
        ax[4,1].plot(t_gt, u_fb_gt[:, 1], label="lcg - gt")
        ax[4,1].plot(t_gt, fb_wb_selected[:, 1], label="lcg - wb")
        ax[4,1].legend()
        ax[4,2].plot(t_gt, u_fb_gt[:, 2], label="ds - gt")
        ax[4,2].plot(t_gt, fb_wb_selected[:, 2], label="ds - wb")
        ax[4,2].legend()
        ax[5,0].plot(t_gt, u_fb_gt[:, 3], label="dr - gt")
        ax[5,0].plot(t_gt, fb_wb_selected[:, 3], label="dr - wb")
        ax[5,0].legend()
        ax[5,1].plot(t_gt, u_cmd_gt[:, 4], label="rpm1 - gt")
        ax[5,1].plot(t_gt, fb_wb_selected[:, 4], label="rpm1 - wb")
        ax[5,1].legend()
        ax[5,2].plot(t_gt, u_cmd_gt[:, 5], label="rpm2 - gt")
        ax[5,2].plot(t_gt, fb_wb_selected[:, 5], label="rpm2 - wb")
        ax[5,2].legend()
        ax[4,0].set_ylabel('u_vbs')
        ax[4,1].set_ylabel('u_lcg')
        ax[4,2].set_ylabel('u_ds')
        ax[5,0].set_ylabel('u_dr')
        ax[5,1].set_ylabel('rpm1')
        ax[5,2].set_ylabel('rpm2')

    # Plotting forces
    if True:
        _, ax = plt.subplots(6, 2, figsize=(12, 10))
        Dv_comp_gt = np.array(Dv_comp_gt)
        Dv_wb_selected = np.array(Dv_wb_selected)
        ax[0,0].plot(t_gt, Dv_comp_gt[:, 0], label="Damping x - gt")
        ax[0,0].plot(t_gt, Dv_wb_selected[:, 0], label="Damping x - wb")
        ax[0,0].legend()
        ax[1,0].plot(t_gt, Dv_comp_gt[:, 1], label="Damping y - gt")
        ax[1,0].plot(t_gt, Dv_wb_selected[:, 1], label="Damping y - wb")
        ax[1,0].legend()
        ax[2,0].plot(t_gt, Dv_comp_gt[:, 2], label="Damping z - gt")
        ax[2,0].plot(t_gt, Dv_wb_selected[:, 2], label="Damping z - wb")
        ax[2,0].legend()
        ax[3,0].plot(t_gt, Dv_comp_gt[:, 3], label="Damping roll - gt")
        ax[3,0].plot(t_gt, Dv_wb_selected[:, 3], label="Damping roll - wb")
        ax[3,0].legend()
        ax[4,0].plot(t_gt, Dv_comp_gt[:, 4], label="Damping pitch - gt")
        ax[4,0].plot(t_gt, Dv_wb_selected[:, 4], label="Damping pitch - wb")
        ax[4,0].legend()
        ax[5,0].plot(t_gt, Dv_comp_gt[:, 5], label="Damping yaw - gt")
        ax[5,0].plot(t_gt, Dv_wb_selected[:, 5], label="Damping yaw - wb")
        ax[5,0].legend()
        ax[0,0].set_ylabel('D*v - x')
        ax[1,0].set_ylabel('D*v - y')
        ax[2,0].set_ylabel('D*v - z')
        ax[3,0].set_ylabel('D*v - p')
        ax[4,0].set_ylabel('D*v - q')
        ax[5,0].set_ylabel('D*v - r')

    # Plotting sim forces in n-axis
    if True:
        
        _, ax = plt.subplots(5, 1, figsize=(12, 10))
        
        sim_length = np.shape(x_acc_wb)[0]
        sim_end = sim_length * dt_model
        t_sim = np.linspace(0, sim_end, sim_length)

        x_acc_wb = np.array(x_acc_wb)
        
        ax[0].plot(t_sim, x_acc_wb[:, 0], label="nu_dot - wb")
        ax[0].legend()
        ax[1].plot(t_sim, x_acc_wb[:, 1], label="C(v)v - wb")
        ax[1].legend()
        ax[2].plot(t_sim, x_acc_wb[:, 2], label="D(v)v - wb")
        ax[2].legend()
        ax[3].plot(t_sim, x_acc_wb[:, 3], label="Tau - wb")
        ax[3].legend()
        ax[4].plot(t_sim, x_acc_wb[:, 4], label="g_vec - wb")
        ax[4].legend()
        ax[0].set_ylabel('nu_dot')
        ax[1].set_ylabel('C(v)v')
        ax[2].set_ylabel('D(v)v')
        ax[3].set_ylabel('Tau')
        ax[4].set_ylabel('g_vec')
   
    # Plotting unsynched sim results
    if True:
        sim_length = np.shape(eta_wb)[0]
        sim_end = sim_length * dt_model
        t_sim = np.linspace(0, sim_end, sim_length)
        eta_wb = np.array(eta_wb)

        _, ax = plt.subplots(6, 3, figsize=(12, 10))

        ax[0,0].plot(t_sim, eta_wb[:, 0], label="x - wb")
        ax[0,0].legend()
        ax[0,1].plot(t_sim, eta_wb[:, 1], label="y - wb")
        ax[0,1].legend()
        ax[0,2].plot(t_sim, eta_wb[:, 2], label="z - wb")
        ax[0,2].legend()
        ax[1,0].plot(t_sim, eta_wb[:, 5], label="roll - wb")
        ax[1,0].legend()
        ax[1,1].plot(t_sim, eta_wb[:, 4], label="pitch - wb")
        ax[1,1].legend()
        ax[1,2].plot(t_sim, eta_wb[:, 3], label="yaw - wb")
        ax[1,2].legend()
        ax[0,0].set_ylabel('x Position [m]')
        ax[0,1].set_ylabel('y Position [m]')
        ax[0,2].set_ylabel('-z Position [m]')
        ax[1,0].set_ylabel('roll [deg]')
        ax[1,1].set_ylabel('pitch [deg]')
        ax[1,2].set_ylabel('yaw [deg]')
        ax[2,0].plot(t_sim, nu_wb[:, 0], label="u - wb")
        ax[2,0].legend()
        ax[2,1].plot(t_sim, nu_wb[:, 1], label="v - wb")
        ax[2,1].legend()
        ax[2,2].plot(t_sim, nu_wb[:, 2], label="w - wb")
        ax[2,2].legend()
        ax[3,0].plot(t_sim, nu_wb[:, 3], label="p - wb")
        ax[3,0].legend()
        ax[3,1].plot(t_sim, nu_wb[:, 4], label="q - wb")
        ax[3,1].legend()
        ax[3,2].plot(t_sim, nu_wb[:, 5], label="r - wb")
        ax[3,2].legend()
        ax[2,0].set_ylabel('u (x_dot)')
        ax[2,1].set_ylabel('v (y_dot)')
        ax[2,2].set_ylabel('w (z_dot)')
        ax[3,0].set_ylabel('p (roll_dot)')
        ax[3,1].set_ylabel('q (pitch_dot)')
        ax[3,2].set_ylabel('r (yaw_dot)')
        ax[4,0].plot(t_sim, fb_wb[:, 0], label="vbs - wb")
        ax[4,0].legend()
        ax[4,1].plot(t_sim, fb_wb[:, 1], label="lcg - wb")
        ax[4,1].legend()
        ax[4,2].plot(t_sim, fb_wb[:, 2], label="ds - wb")
        ax[4,2].legend()
        ax[5,0].plot(t_sim, fb_wb[:, 3], label="dr - wb")
        ax[5,0].legend()
        ax[5,1].plot(t_sim, fb_wb[:, 4], label="rpm1 - wb")
        ax[5,1].legend()
        ax[5,2].plot(t_sim, fb_wb[:, 5], label="rpm2 - wb")
        ax[5,2].legend()
        ax[4,0].set_ylabel('u_vbs')
        ax[4,1].set_ylabel('u_lcg')
        ax[4,2].set_ylabel('u_ds')
        ax[5,0].set_ylabel('u_dr')
        ax[5,1].set_ylabel('rpm1')
        ax[5,2].set_ylabel('rpm2')


    plt.show()



# TODO: Put this in later
# def rk4(x, u, dt, fun):
#     k1 = fun(x, u)
#     k2 = fun(x+dt/2*k1, u)
#     k3 = fun(x+dt/2*k2, u)
#     k4 = fun(x+dt*k3, u)

#     x_t = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

#     return x_t        