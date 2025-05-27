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

    def __init__(self, model_type: str, init_pose: list, time_vec: list, control_vec: list, vehicle: str):

        # Initial conditions 
        self.x0 = init_pose

        # Create vehicle instance
        if vehicle == "SAM":
            self.vehicle = SAM_PIML(dt=0.01, piml_type=model_type)
        elif vehicle == "BROV":
            self.vehicle = BlueROV_PIML(h=0.01, piml_type=model_type)
        else: 
            print("Selected vehicle for VEHICLE_SIM does not exist")
            return
        
        # Calculating how many sim steps we need to cover full time
        self.controls = control_vec
        self.n_sim = np.shape(time_vec)[0]
        self.var_dt = np.diff(time_vec)

    def run_sim(self):
        
        print(f" Running simulator...")
        # For storing results of sim
        data = np.empty((len(self.x0), self.n_sim))
        data[:,0] = self.x0

        # Extra outputs for controlling results
        Dv_vec = []
        x_acc_vec = []

        # Integration loop
        for i in range(self.n_sim-1):

            # Get the timestep
            dt = self.var_dt[i]
            self.vehicle.update_dt(dt)

            # Try to do a normal sim step
            # try:
            data[:,i+1], Dv, x_acc = self.rk4(data[:,i], self.controls[i], dt, self.vehicle.dynamics)
            Dv_vec.append(Dv)
            x_acc_vec.append(x_acc)

            # Otherwise normalize the quat and try again
            # except:
            #     # TODO: By the points we get here the speeds are already way to big, maybe if we fix the other sim problems this will not be an issue(?)
            #     # NOTE: This issue happens more due to the fact that we have an overflow in the speeds turning them into NaNs --> making the quats weird --> crashing the sim
            #     # NOTE: So if we actually want to fix this just check for NaNs in the update and replace with 0 or whatever you want
               
            #     # Normalize quaternion to help with stability when sim is misbehaving
            #     quat = data[:, i][2:6]
            #     quat = quat / np.linalg.norm(quat)
            #     data[:, i][3] = quat[0]
            #     data[:, i][4] = quat[1]
            #     data[:, i][5] = quat[2]
            #     data[:, i][6] = quat[3]
                
            #     data[:,i+1], Dv, x_acc = self.rk4(data[:,i], self.controls[i], dt, self.vehicle.dynamics)
            #     Dv_vec.append(Dv)
            #     x_acc_vec.append(x_acc)

            #     # If this doesn't work we just let the sim crash atm (useful for PIML validation of D)

        return data, Dv_vec, x_acc_vec
    
    def rk4(self, x, u, dt, fun):
        # https://github.com/smarc-project/smarc_modelling/blob/master/src/smarc_modelling/sam_sim.py#L38C1-L46C15 
        # Runge Kutta 4
        k1, Dv, x_acc = fun(x, u) # (19, )
        k2, _, _ = fun(x+dt/2*k1, u)
        k3, _, _ = fun(x+dt/2*k2, u)
        k4, _, _ = fun(x+dt*k3, u)
        x_t = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x_t, Dv, x_acc
    
    def ef(self, x, u, dt, fun):
        # Euler forward
        dx, Dv, x_acc = fun(x, u)
        x_t = x + dx * dt
        return x_t, Dv, x_acc


if __name__ == "__main__":
    print(f" Initializing simulator...")
    
    # Ground truth data (SAM)
    # eta_gt, nu_gt, u_fb_gt, u_cmd_gt, Dv_comp_gt, Mv_dot_gt, Cv_gt, g_eta_gt, tau_gt, t_gt = load_data_from_bag("src/smarc_modelling/piml/data/rosbags/rosbag_tank_validate", "torch")
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
    # u_cmd_gt[:, 3] = 0 #-np.deg2rad(7)   # Horizontal (rudder)
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

    # Simulator parameters (Not used now but its a nice print out)
    dt = np.mean(np.diff(t_gt)) # Time step 
    print(f" Dataset average dt as: {dt}")

    # Models for simulation
    print(f" Initalizing models...")
    sam_wb = VEHICLE_SIM(None, init_pose, t_gt, u_cmd_gt, "SAM")
    # sam_pinn = VEHICLE_SIM("pinn", dt_model, init_pose, t_gt, u_cmd_gt, "SAM")

    # Running simulations
    print(f" Running all the simulators...")

    # White-box
    results_wb, Dv_wb, acc_wb = sam_wb.run_sim()
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

    # Convert quats to angles for interpretation in plots
    eta_gt = [eta_quat_to_deg(eta_vec) for eta_vec in eta_gt] 
    eta_wb = [eta_quat_to_deg(eta_vec) for eta_vec in eta_wb] 

    # Turn torch/list into np arrays and adjust gt size
    eta_gt = np.array(eta_gt)
    nu_gt = np.array(nu_gt)
    t_gt = np.array(t_gt)

    eta_wb = np.array(eta_wb)
    nu_wb = np.array(nu_wb)
    fb_wb = np.array(fb_wb)

    # Plotting each state
    if True:
        _, ax = plt.subplots(6, 3, figsize=(12, 10))

        # Pose
        ax[0,0].plot(t_gt, eta_gt[:, 0], label="x - gt")
        ax[0,0].plot(t_gt, eta_wb[:, 0], label="x - wb")
        ax[0,0].legend()
        ax[0,1].plot(t_gt, eta_gt[:, 1], label="y - gt")
        ax[0,1].plot(t_gt, eta_wb[:, 1], label="y - wb")
        ax[0,1].legend()
        ax[0,2].plot(t_gt, eta_gt[:, 2], label="z - gt")
        ax[0,2].plot(t_gt, eta_wb[:, 2], label="z - wb")
        ax[0,2].legend()
        ax[1,0].plot(t_gt, eta_gt[:, 5], label="roll - gt")
        ax[1,0].plot(t_gt, eta_wb[:, 5], label="roll - wb")
        ax[1,0].legend()
        ax[1,1].plot(t_gt, eta_gt[:, 4], label="pitch - gt")
        ax[1,1].plot(t_gt, eta_wb[:, 4], label="pitch - wb")
        ax[1,1].legend()
        ax[1,2].plot(t_gt, eta_gt[:, 3], label="yaw - gt")
        ax[1,2].plot(t_gt, eta_wb[:, 3], label="yaw - wb")
        ax[1,2].legend()
        ax[0,0].set_ylabel('x Position [m]')
        ax[0,1].set_ylabel('y Position [m]')
        ax[0,2].set_ylabel('-z Position [m]')
        ax[1,0].set_ylabel('roll [deg]')
        ax[1,1].set_ylabel('pitch [deg]')
        ax[1,2].set_ylabel('yaw [deg]')

        # Velocities
        ax[2,0].plot(t_gt, nu_gt[:, 0], label="u - gt")
        ax[2,0].plot(t_gt, nu_wb[:, 0], label="u - wb")
        ax[2,0].legend()
        ax[2,1].plot(t_gt, nu_gt[:, 1], label="v - gt")
        ax[2,1].plot(t_gt, nu_wb[:, 1], label="v - wb")
        ax[2,1].legend()
        ax[2,2].plot(t_gt, nu_gt[:, 2], label="w - gt")
        ax[2,2].plot(t_gt, nu_wb[:, 2], label="w - wb")
        ax[2,2].legend()
        ax[3,0].plot(t_gt, nu_gt[:, 3], label="p - gt")
        ax[3,0].plot(t_gt, nu_wb[:, 3], label="p - wb")
        ax[3,0].legend()
        ax[3,1].plot(t_gt, nu_gt[:, 4], label="q - gt")
        ax[3,1].plot(t_gt, nu_wb[:, 4], label="q - wb")
        ax[3,1].legend()
        ax[3,2].plot(t_gt, nu_gt[:, 5], label="r - gt")
        ax[3,2].plot(t_gt, nu_wb[:, 5], label="r - wb")
        ax[3,2].legend()
        ax[2,0].set_ylabel('u (x_dot)')
        ax[2,1].set_ylabel('v (y_dot)')
        ax[2,2].set_ylabel('w (z_dot)')
        ax[3,0].set_ylabel('p (roll_dot)')
        ax[3,1].set_ylabel('q (pitch_dot)')
        ax[3,2].set_ylabel('r (yaw_dot)')

        # Controls
        ax[4,0].plot(t_gt, u_fb_gt[:, 0], label="vbs - gt")
        ax[4,0].plot(t_gt, fb_wb[:, 0], label="vbs - wb")
        ax[4,0].legend()
        ax[4,1].plot(t_gt, u_fb_gt[:, 1], label="lcg - gt")
        ax[4,1].plot(t_gt, fb_wb[:, 1], label="lcg - wb")
        ax[4,1].legend()
        ax[4,2].plot(t_gt, u_fb_gt[:, 2], label="ds - gt")
        ax[4,2].plot(t_gt, fb_wb[:, 2], label="ds - wb")
        ax[4,2].legend()
        ax[5,0].plot(t_gt, u_fb_gt[:, 3], label="dr - gt")
        ax[5,0].plot(t_gt, fb_wb[:, 3], label="dr - wb")
        ax[5,0].legend()
        ax[5,1].plot(t_gt, u_cmd_gt[:, 4], label="rpm1 - gt")
        ax[5,1].plot(t_gt, fb_wb[:, 4], label="rpm1 - wb")
        ax[5,1].legend()
        ax[5,2].plot(t_gt, u_cmd_gt[:, 5], label="rpm2 - gt")
        ax[5,2].plot(t_gt, fb_wb[:, 5], label="rpm2 - wb")
        ax[5,2].legend()
        ax[4,0].set_ylabel('u_vbs')
        ax[4,1].set_ylabel('u_lcg')
        ax[4,2].set_ylabel('u_ds')
        ax[5,0].set_ylabel('u_dr')
        ax[5,1].set_ylabel('rpm1')
        ax[5,2].set_ylabel('rpm2')
        # Limit for removing the spike
        ax[4,2].set_ylim(-10, 10)
        ax[5,0].set_ylim(-10, 10)
        ax[5,1].set_ylim(-1000, 1000)
        ax[5,2].set_ylim(-1000, 1000)

    # Plotting forces
    if True:
        _, ax = plt.subplots(6, 2, figsize=(12, 10))
        Dv_comp_gt = np.array(Dv_comp_gt)
        Dv_wb = np.array(Dv_wb)

        t_gt_forces = t_gt[:-1]
        Dv_comp_gt = Dv_comp_gt[:-1]

        ax[0,0].plot(t_gt_forces, Dv_comp_gt[:, 0], label="Damping x - gt")
        ax[0,0].plot(t_gt_forces, Dv_wb[:, 0], label="Damping x - wb")
        ax[0,0].legend()
        ax[1,0].plot(t_gt_forces, Dv_comp_gt[:, 1], label="Damping y - gt")
        ax[1,0].plot(t_gt_forces, Dv_wb[:, 1], label="Damping y - wb")
        ax[1,0].legend()
        ax[2,0].plot(t_gt_forces, Dv_comp_gt[:, 2], label="Damping z - gt")
        ax[2,0].plot(t_gt_forces, Dv_wb[:, 2], label="Damping z - wb")
        ax[2,0].legend()
        ax[3,0].plot(t_gt_forces, Dv_comp_gt[:, 3], label="Damping roll - gt")
        ax[3,0].plot(t_gt_forces, Dv_wb[:, 3], label="Damping roll - wb")
        ax[3,0].legend()
        ax[4,0].plot(t_gt_forces, Dv_comp_gt[:, 4], label="Damping pitch - gt")
        ax[4,0].plot(t_gt_forces, Dv_wb[:, 4], label="Damping pitch - wb")
        ax[4,0].legend()
        ax[5,0].plot(t_gt_forces, Dv_comp_gt[:, 5], label="Damping yaw - gt")
        ax[5,0].plot(t_gt_forces, Dv_wb[:, 5], label="Damping yaw - wb")
        ax[5,0].legend()
        ax[0,0].set_ylabel('D*v - x')
        ax[1,0].set_ylabel('D*v - y')
        ax[2,0].set_ylabel('D*v - z')
        ax[3,0].set_ylabel('D*v - p')
        ax[4,0].set_ylabel('D*v - q')
        ax[5,0].set_ylabel('D*v - r')

    # Plotting sim forces in n-axis
    if True:
        
        _, ax = plt.subplots(5, 6, figsize=(12, 10))
   
        acc_wb = np.array(acc_wb)
        x_acc_wb = np.array(acc_wb[:, :, 0])
        y_acc_wb = np.array(acc_wb[:, :, 1])
        z_acc_wb = np.array(acc_wb[:, :, 2])
        p_acc_wb = np.array(acc_wb[:, :, 3])
        q_acc_wb = np.array(acc_wb[:, :, 4])
        r_acc_wb = np.array(acc_wb[:, :, 5])

        t_gt_forces = t_gt[:-1]
        
        ax[0, 0].plot(t_gt_forces, x_acc_wb[:, 0], label="nu_dot - wb - x")
        ax[0, 1].plot(t_gt_forces, y_acc_wb[:, 0], label="nu_dot - wb - y")
        ax[0, 2].plot(t_gt_forces, z_acc_wb[:, 0], label="nu_dot - wb - z")
        ax[0, 3].plot(t_gt_forces, p_acc_wb[:, 0], label="nu_dot - wb - p")
        ax[0, 4].plot(t_gt_forces, q_acc_wb[:, 0], label="nu_dot - wb - q")
        ax[0, 5].plot(t_gt_forces, r_acc_wb[:, 0], label="nu_dot - wb - r")
        # ax[0].legend()
        ax[1, 0].plot(t_gt_forces, x_acc_wb[:, 1], label="C(v)v - wb - x")
        ax[1, 1].plot(t_gt_forces, y_acc_wb[:, 1], label="C(v)v - wb - y")
        ax[1, 2].plot(t_gt_forces, z_acc_wb[:, 1], label="C(v)v - wb - z")
        ax[1, 3].plot(t_gt_forces, p_acc_wb[:, 1], label="C(v)v - wb - p")
        ax[1, 4].plot(t_gt_forces, q_acc_wb[:, 1], label="C(v)v - wb - q")
        ax[1, 5].plot(t_gt_forces, r_acc_wb[:, 1], label="C(v)v - wb - r")
        # ax[1].legend()
        ax[2, 0].plot(t_gt_forces, x_acc_wb[:, 2], label="D(v)v - wb - x")
        ax[2, 1].plot(t_gt_forces, y_acc_wb[:, 2], label="D(v)v - wb - y")
        ax[2, 2].plot(t_gt_forces, z_acc_wb[:, 2], label="D(v)v - wb - z")
        ax[2, 3].plot(t_gt_forces, p_acc_wb[:, 2], label="D(v)v - wb - p")
        ax[2, 4].plot(t_gt_forces, q_acc_wb[:, 2], label="D(v)v - wb - q")
        ax[2, 5].plot(t_gt_forces, r_acc_wb[:, 2], label="D(v)v - wb - r")
        # ax[2].legend()
        ax[3, 0].plot(t_gt_forces, x_acc_wb[:, 3], label="Tau - wb - x")
        ax[3, 1].plot(t_gt_forces, y_acc_wb[:, 3], label="Tau - wb - y")
        ax[3, 2].plot(t_gt_forces, z_acc_wb[:, 3], label="Tau - wb - z")
        ax[3, 3].plot(t_gt_forces, p_acc_wb[:, 3], label="Tau - wb - p")
        ax[3, 4].plot(t_gt_forces, q_acc_wb[:, 3], label="Tau - wb - q")
        ax[3, 5].plot(t_gt_forces, r_acc_wb[:, 3], label="Tau - wb - r")
        # ax[3].legend()
        ax[4, 0].plot(t_gt_forces, x_acc_wb[:, 4], label="g_vec - wb - x")
        ax[4, 1].plot(t_gt_forces, y_acc_wb[:, 4], label="g_vec - wb - y")
        ax[4, 2].plot(t_gt_forces, z_acc_wb[:, 4], label="g_vec - wb - z")
        ax[4, 3].plot(t_gt_forces, p_acc_wb[:, 4], label="g_vec - wb - p")
        ax[4, 4].plot(t_gt_forces, q_acc_wb[:, 4], label="g_vec - wb - q")
        ax[4, 5].plot(t_gt_forces, r_acc_wb[:, 4], label="g_vec - wb - r")
        # ax[4].legend()
        ax[0, 0].set_ylabel('nu_dot')
        ax[1, 0].set_ylabel('D(v)v')
        ax[2, 0].set_ylabel('C(v)v')
        ax[3, 0].set_ylabel('Tau')
        ax[4, 0].set_ylabel('g_vec')

        ax[0, 0].set_xlabel('x')
        ax[0, 1].set_xlabel('y')
        ax[0, 2].set_xlabel('z')
        ax[0, 3].set_xlabel('p')
        ax[0, 4].set_xlabel('q')
        ax[0, 5].set_xlabel('r')
   
    # Plotting unsynched sim results
    if True:

        eta_wb = np.array(eta_wb)

        _, ax = plt.subplots(6, 3, figsize=(12, 10))

        ax[0,0].plot(t_gt, eta_wb[:, 0], label="x - wb")
        ax[0,0].legend()
        ax[0,1].plot(t_gt, eta_wb[:, 1], label="y - wb")
        ax[0,1].legend()
        ax[0,2].plot(t_gt, eta_wb[:, 2], label="z - wb")
        ax[0,2].legend()
        ax[1,0].plot(t_gt, eta_wb[:, 5], label="roll - wb")
        ax[1,0].legend()
        ax[1,1].plot(t_gt, eta_wb[:, 4], label="pitch - wb")
        ax[1,1].legend()
        ax[1,2].plot(t_gt, eta_wb[:, 3], label="yaw - wb")
        ax[1,2].legend()
        ax[0,0].set_ylabel('x Position [m]')
        ax[0,1].set_ylabel('y Position [m]')
        ax[0,2].set_ylabel('-z Position [m]')
        ax[1,0].set_ylabel('roll [deg]')
        ax[1,1].set_ylabel('pitch [deg]')
        ax[1,2].set_ylabel('yaw [deg]')
        ax[2,0].plot(t_gt, nu_wb[:, 0], label="u - wb")
        ax[2,0].legend()
        ax[2,1].plot(t_gt, nu_wb[:, 1], label="v - wb")
        ax[2,1].legend()
        ax[2,2].plot(t_gt, nu_wb[:, 2], label="w - wb")
        ax[2,2].legend()
        ax[3,0].plot(t_gt, nu_wb[:, 3], label="p - wb")
        ax[3,0].legend()
        ax[3,1].plot(t_gt, nu_wb[:, 4], label="q - wb")
        ax[3,1].legend()
        ax[3,2].plot(t_gt, nu_wb[:, 5], label="r - wb")
        ax[3,2].legend()
        ax[2,0].set_ylabel('u (x_dot)')
        ax[2,1].set_ylabel('v (y_dot)')
        ax[2,2].set_ylabel('w (z_dot)')
        ax[3,0].set_ylabel('p (roll_dot)')
        ax[3,1].set_ylabel('q (pitch_dot)')
        ax[3,2].set_ylabel('r (yaw_dot)')
        ax[4,0].plot(t_gt, fb_wb[:, 0], label="vbs - wb")
        ax[4,0].legend()
        ax[4,1].plot(t_gt, fb_wb[:, 1], label="lcg - wb")
        ax[4,1].legend()
        ax[4,2].plot(t_gt, fb_wb[:, 2], label="ds - wb")
        ax[4,2].legend()
        ax[5,0].plot(t_gt, fb_wb[:, 3], label="dr - wb")
        ax[5,0].legend()
        ax[5,1].plot(t_gt, fb_wb[:, 4], label="rpm1 - wb")
        ax[5,1].legend()
        ax[5,2].plot(t_gt, fb_wb[:, 5], label="rpm2 - wb")
        ax[5,2].legend()
        ax[4,0].set_ylabel('u_vbs')
        ax[4,1].set_ylabel('u_lcg')
        ax[4,2].set_ylabel('u_ds')
        ax[5,0].set_ylabel('u_dr')
        ax[5,1].set_ylabel('rpm1')
        ax[5,2].set_ylabel('rpm2')

    plt.show()
