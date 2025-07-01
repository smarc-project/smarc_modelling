#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smarc_modelling.piml.bpinn.bpinn import BPINN, loss_function
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_deg
from smarc_modelling.piml.piml_sim import SIM
import torch
import numpy as np

test_datasets = ["test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7", "test_8"]

datasets = ["rosbag_12", "rosbag_1", "rosbag_2", "rosbag_3", "rosbag_4", "rosbag_5", "rosbag_6", 
            "rosbag_7", "rosbag_8", "rosbag_9", "rosbag_10", "rosbag_11"]

if __name__ == "__main__":

    # Divide up the datasets into training and validation
    train_procent = 0.8 # How much goes to training, the rest goes to validation
    train_val_split = int(np.shape(datasets)[0] * train_procent)

    # Loading the training data
    x_trajectories = []
    y_trajectories = []

    for dataset in datasets[:train_val_split]:
        # Load data from bag
        path = "src/smarc_modelling/piml/data/rosbags/" + dataset
        eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t = load_data_from_bag(path, "torch")
        
        x_traj = torch.cat([eta, nu, u], dim=1)
        y_traj = {
            "u_cmd": u_cmd,
            "Dv_comp": Dv_comp,
            "Mv_dot": Mv_dot,
            "Cv": Cv,
            "g_eta": g_eta,
            "tau": tau,
            "t": t,
            "nu": nu
        }

        # Append most recently loaded data
        x_trajectories.append(x_traj)
        y_trajectories.append(y_traj)

    # Loading the validation data
    x_trajectories_val = []
    y_trajectories_val = []

    for dataset in datasets[train_val_split:]:
        # Load data from bag
        path = "src/smarc_modelling/piml/data/rosbags/" + dataset
        eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t = load_data_from_bag(path, "torch")
        
        x_traj = torch.cat([eta, nu, u], dim=1)
        y_traj = {
            "u_cmd": u_cmd,
            "Dv_comp": Dv_comp,
            "Mv_dot": Mv_dot,
            "Cv": Cv,
            "g_eta": g_eta,
            "tau": tau,
            "t": t,
            "nu": nu
        }

        # Append most recently loaded data
        x_trajectories_val.append(x_traj)
        y_trajectories_val.append(y_traj)

    # Loading the testing data
    x_trajectories_test = []
    y_trajectories_test = []

    for dataset in test_datasets:
        # Load data from bag
        path = "src/smarc_modelling/piml/data/rosbags/" + dataset
        eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t = load_data_from_bag(path, "torch")

        x_traj = torch.cat([eta, nu, nu], dim=1)
        y_traj = {
            "u_cmd": u_cmd,
            "Dv_comp": Dv_comp,
            "Mv_dot": Mv_dot,
            "Cv": Cv,
            "g_eta": g_eta,
            "tau": tau,
            "t": t,
            "nu": nu,
            "eta": eta
        }

        # Append most recently loaded data
        x_trajectories_test.append(x_traj)
        y_trajectories_test.append(y_traj)

    # For results
    error_grid = np.zeros((5, 4, 5))
    best_error = float("inf")

    # Grid params
    for i, layers in enumerate([5, 8, 12, 15, 20]): # Amount of layers
        for j, size in enumerate([16, 32, 64, 128]): # Amount of neurons in each layer 
            for k, dropout_rate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]): # Starting learning rate <-- Changed by the scheduler during training later

                # NN shape
                shape = [size] * layers # Hidden layers
                shape.insert(0, 19) # Input layer
                shape.append(36) # Output layer
                
                # Initalize model
                model = BPINN(shape, dropout_rate)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

                # Adaptive learning rate
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9, patience=500, threshold=0.01, min_lr=1e-5)

                # Early stopping with validation loss
                best_val_loss = float("inf")
                patience = 7500
                counter = 0
                best_model_state = None

                # Training loop
                epochs = 500000
                for epoch in range(epochs):

                    model.train()
                    optimizer.zero_grad()

                    # Forward pass for each of our trajectories
                    loss_total = 0.0
                    for x_traj, y_traj in zip(x_trajectories, y_trajectories):
                        
                        # Get PI loss
                        loss = loss_function(model, x_traj, y_traj["Dv_comp"], y_traj["Mv_dot"], y_traj["Cv"], y_traj["g_eta"], y_traj["tau"], y_traj["nu"])
                        loss_total += loss

                    loss_total.backward()
                    optimizer.step()

                    # Step scheduler
                    scheduler.step(loss_total)

                    # Evaluate model on validation data
                    model.eval()
                    val_loss_total = 0.0
                    with torch.no_grad():
                        for x_traj_val, y_traj_val in zip(x_trajectories_val, y_trajectories_val):

                            # Get PI validation loss
                            val_loss = loss_function(model, x_traj_val, y_traj_val["Dv_comp"], y_traj_val["Mv_dot"], y_traj_val["Cv"], y_traj_val["g_eta"], y_traj_val["tau"], y_traj_val["nu"])
                            val_loss_total += val_loss
                        
                    # Handling for early stopping
                    if val_loss_total.item() < best_val_loss:
                        best_val_loss = val_loss_total.item()
                        counter = 0
                        best_model_state = model.state_dict()
                    else:
                        counter += 1

                    if epoch % 500 == 0:
                        print(f" Still training, epoch {epoch}, loss: {loss_total.item()}, lr: {optimizer.param_groups[0]['lr']},\n validation loss: {val_loss_total.item()}, shape: {layers}, {size}, {dropout_rate}")

                    if counter >= patience:
                        print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}")
                        break
                
                # Calculating the model error
                model.load_state_dict(best_model_state) # Loading the best model state
                torch.save({"model_shape": shape, "state_dict": model.state_dict(), "dropout": dropout_rate}, "src/smarc_modelling/piml/models/bpinn.pt") 
                model.eval() # Just to doubly ensure that it is in eval mode

                total_error = 0.0
                for x_traj_test, y_traj_test in zip(x_trajectories_test, y_trajectories_test):

                    # For flipping the sim results later
                    x0 = y_traj_test["eta"][0, 0].item()
                    y0 = y_traj_test["eta"][0, 1].item()
                    z0 = y_traj_test["eta"][0, 2].item()
                    eta_for_error = y_traj_test["eta"]
                    eta_for_error[:, 1] = 2 * y0 - eta_for_error[:, 1]
                    eta_for_error[:, 2] = 2 * z0 - eta_for_error[:, 2]
                    eta_test_degs = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_for_error])

                    try: 
                        # Running the SAM simulator to get predicted validation path
                        sam_pinn = SIM("bpinn", x_traj_test[0], y_traj_test["t"], y_traj_test["u_cmd"])
                        results, _ = sam_pinn.run_sim()
                        results = torch.tensor(results).T
                        eta_pinn = results[:, 0:7]
                        eta_pinn[:, 0] = 2 * x0 - eta_pinn[:, 0] # Flipping to NED frame
                        eta_pinn[:, 2] = 2 * z0 - eta_pinn[:, 2]
                        nu_pinn = results[:, 7:13]

                        # Convert quat to angles
                        eta_pinn_degs = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_pinn])

                        # Calculated summed error
                        eta_mse = np.array((eta_pinn_degs - eta_test_degs)**2)
                        nu_mse = np.array((nu_pinn - y_traj_test["nu"])**2)
                        error = np.sum(eta_mse) + np.sum(nu_mse)

                        total_error += error
                        print(f" Test passed successfully with error: {error}. \n")

                    except Exception as e:
                        # Since many of the models will be bad from the grid training,
                        # they will lead to the simulator going to inf and breaking it so we need to 
                        # have an except for these cases
                        print(f" {e}")
                        total_error += 10e10 # Big number to penalize bad models

                error_grid[i, j, k] = total_error

                # Save the best model for later
                if total_error < best_error:
                            best_error = total_error
                            torch.save({"model_shape": shape, "state_dict": model.state_dict(), "dropout": dropout_rate}, "src/smarc_modelling/piml/models/bpinn_best_grid.pt")
                            best_setup = [layers, size, dropout_rate]
    
                
    # After going trough the grid getting the smallest loss
    print(f" Best found configuration as: {best_setup}")
    print(f" Training set was: {datasets[:train_val_split]}")
    print(f" Validation set was: {datasets[train_val_split:]}")

