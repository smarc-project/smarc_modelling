#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smarc_modelling.piml.pinn.pinn import PINN, loss_function
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_deg
from smarc_modelling.piml.piml_sim import SIM
import torch
import numpy as np
import matplotlib.pyplot as plt

test_datasets = ["test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7", "test_8"]

datasets = ["rosbag_16", "rosbag_15", "rosbag_14", "rosbag_13", "rosbag_12", "rosbag_1", "rosbag_2", "rosbag_3", "rosbag_4", "rosbag_5", "rosbag_6", 
            "rosbag_7", "rosbag_8", "rosbag_9", "rosbag_10", "rosbag_11"]

if __name__ == "__main__":

    # Divide up the datasets into training and validation
    train_procent = 0.9 # How much goes to training, the rest goes to validation
    train_val_split = int(np.shape(datasets)[0] * train_procent)

    # Loading the training data
    x_trajectories = []
    y_trajectories = []

    for dataset in datasets[:train_val_split]:
        # Load data from bag
        path = "src/smarc_modelling/piml/data/rosbags/" + dataset
        eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, acc = load_data_from_bag(path, "torch")
        
        x_traj = torch.cat([eta, nu, u], dim=1)
        y_traj = {
            "u_cmd": u_cmd,
            "Dv_comp": Dv_comp,
            "Mv_dot": Mv_dot,
            "Cv": Cv,
            "g_eta": g_eta,
            "tau": tau,
            "t": t,
            "nu": nu,
            "M": M,
            "acc": acc
        }

        # Append most recently loaded data
        x_trajectories.append(x_traj)
        y_trajectories.append(y_traj)

    # Normalize the data
    all_x = torch.cat(x_trajectories, dim=0)
    x_mean = torch.mean(all_x, dim=0)
    x_std = torch.std(all_x, dim=0) + 1e-8 # Preventing division by 0

    # Loading the validation data
    x_trajectories_val = []
    y_trajectories_val = []

    for dataset in datasets[train_val_split:]:
        # Load data from bag
        path = "src/smarc_modelling/piml/data/rosbags/" + dataset
        eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, acc = load_data_from_bag(path, "torch")
        
        x_traj = torch.cat([eta, nu, u], dim=1)
        y_traj = {
            "u_cmd": u_cmd,
            "Dv_comp": Dv_comp,
            "Mv_dot": Mv_dot,
            "Cv": Cv,
            "g_eta": g_eta,
            "tau": tau,
            "t": t,
            "nu": nu,
            "M": M,
            "acc": acc
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
        eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, acc = load_data_from_bag(path, "torch")

        x_traj = torch.cat([eta, nu, u], dim=1)
        y_traj = {
            "u_cmd": u_cmd,
            "Dv_comp": Dv_comp,
            "Mv_dot": Mv_dot,
            "Cv": Cv,
            "g_eta": g_eta,
            "tau": tau,
            "t": t,
            "nu": nu,
            "eta": eta,
            "M": M,
            "acc": acc
        }

        # Append most recently loaded data
        x_trajectories_test.append(x_traj)
        y_trajectories_test.append(y_traj)

    # For results
    error_grid = np.zeros((6, 6, 3))
    best_error = float("inf")

    # We do not normalize the test dataset since we need the actual values in the sim
    # instead the normalization is done directly before prediction the damping matrix letting us 
    # retain these original values

    n_steps = 20

    best_loss_history = []
    best_val_loss_history = []

    # Grid params
    for i, layers in enumerate([5, 10, 20, 30, 50, 100]): # Amount of layers
        for j, size in enumerate([4, 8, 16, 32, 64, 128]): # Amount of neurons in each layer 
            for k, factor in enumerate([0.75, 0.5, 0.25]):

                loss_history = []
                val_loss_history = []

                # NN shape
                shape = [size] * layers # Hidden layers
                shape.insert(0, 19) # Input layer
                shape.append(36) # Output layer
                
                # Initalize model
                model = PINN(shape)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping to help with stability

                # Adaptive learning rate
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=factor, patience=2500, threshold=10, min_lr=1e-8)

                # Early stopping with validation loss
                best_val_loss = float("inf")
                patience = 5000
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
                        x_traj_normed = (x_traj - x_mean) / x_std
                        loss = loss_function(model, x_traj_normed, y_traj, n_steps)
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
                            x_traj_val_normed = (x_traj_val - x_mean) / x_std
                            val_loss = loss_function(model, x_traj_val_normed, y_traj_val, n_steps)
                            val_loss_total += val_loss

                    # For plotting the loss 
                    loss_history.append(loss_total.item())
                    val_loss_history.append(val_loss_total.item())

                    # Handling for early stopping
                    if val_loss_total.item() < best_val_loss:
                        best_val_loss = val_loss_total.item()
                        counter = 0
                        best_model_state = model.state_dict()
                    else:
                        counter += 1

                    # Print statement to make sure everything is still running
                    if epoch % 500 == 0:
                        print(f" Still training, epoch {epoch}, loss: {loss_total.item()}, lr: {optimizer.param_groups[0]['lr']},\n validation loss: {val_loss_total.item()}, shape: {layers}, {size}, {factor}, counter: {counter}")

                    # Break condition for early stopping
                    if counter >= patience:
                        print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}\n")
                        break
                
                # Calculating the model error
                model.load_state_dict(best_model_state) # Loading the best model state
                torch.save({"model_shape": shape, 
                            "state_dict": model.state_dict(),
                            "x_mean": x_mean,
                            "x_std": x_std}, 
                            "src/smarc_modelling/piml/models/pinn.pt") 
                model.eval() # Just to doubly ensure that it is in eval mode

                total_error = 0.0
                i = 0
                for x_traj_test, y_traj_test in zip(x_trajectories_test, y_trajectories_test):
                    
                    # For flipping the sim results later
                    x0 = y_traj_test["eta"][0, 0].item()
                    y0 = y_traj_test["eta"][0, 1].item()
                    z0 = y_traj_test["eta"][0, 2].item()
                    eta_for_error = y_traj_test["eta"]
                    eta_for_error[:, 1] = 2 * y0 - eta_for_error[:, 1]
                    eta_for_error[:, 2] = 2 * z0 - eta_for_error[:, 2]
                    eta_test_degs = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_for_error])

                    # Getting the state vector in such a way that we can use it in the simulator
                    states_test = [x_traj_test[0:7][:], x_traj_test[7:13][:], x_traj_test[13:19][:]]

                    try: 
                        # Running the SAM simulator to get predicted validation path
                        # In the sim the values are normalized before getting the prediction for the damping matrix
                        
                        sam_pinn = SIM("pinn", states_test, y_traj_test["t"], y_traj_test["u_cmd"], False)
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
                        total_error += 10e10 # Big number to discard bad models

                error_grid[i, j, k] = total_error

                # Save the best model for later
                if total_error < best_error:
                            best_error = total_error
                            torch.save({"model_shape": shape, 
                                        "state_dict": model.state_dict(), 
                                        "x_mean": x_mean,
                                        "x_std": x_std},
                                        "src/smarc_modelling/piml/models/pinn_best_grid.pt")
                            best_setup = [layers, size, factor]

                            best_loss_history = loss_history
                            best_val_loss_history = val_loss_history
    

    # After going trough the grid getting the smallest loss
    print(f" Best found configuration as: {best_setup}")
    print(f" Training set was: {datasets[:train_val_split]}")
    print(f" Validation set was: {datasets[train_val_split:]}")

    # Plotting the loss and lr
    plt.style.use('science')

    # Loss
    ax = plt.subplot(1, 1, 1)
    plt.plot(best_loss_history, linestyle="-", color="green", label="Training loss")
    ax.set_yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("log(Loss)")
    plt.plot(best_val_loss_history, linestyle="-", color="red", label="Validation loss")
    plt.legend()

    # Display
    plt.show()

    # Grid trainer log
    # 6 - 128 best Jun 6 23:12
    # 12 - 32 best Jun 8 13:23
    # 20 - 32 best Jun 8 21:23
    # New additional data added to training set
    # 15 - 16 - 0.9 best Jun 13 20:33
    # Changed to trajectory based training with multiple test cases Jun 23
    # 4 - 6 - 0.9 best Jun 24 10:46 <-- Model isn't that good but it was able to predict the full bag
    # With multi-loss
    # 8 - 16 - 0.9 best Jul 3 12:11
    # 10 - 128 - 0.75 best Jul 8 13:55