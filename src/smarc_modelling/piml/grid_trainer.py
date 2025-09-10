#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smarc_modelling.piml.pinn import PINN
from smarc_modelling.piml.bpinn import BPINN
from smarc_modelling.piml.nn import NN
from smarc_modelling.piml.naive_nn import NaiveNN
from smarc_modelling.piml.utils.utility_functions import load_to_trajectory, eta_quat_to_deg
from smarc_modelling.piml.piml_sim import SIM
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots # For fancy plotting
import random

test_datasets = ["rosbag_3", "rosbag_18", "rosbag_85", "rosbag_112", "rosbag_113", "rosbag_114"]

datasets = ["rosbag_1", "rosbag_5", "rosbag_9", "rosbag_10", "rosbag_11", "rosbag_12", "rosbag_13", 
            "rosbag_15", "rosbag_16", "rosbag_17", "rosbag_19", "rosbag_20", "rosbag_25", "rosbag_28", 
            "rosbag_31", "rosbag_34", "rosbag_38", "rosbag_39", "rosbag_41", "rosbag_43", "rosbag_46",
            "rosbag_47", "rosbag_48", "rosbag_49", "rosbag_50", "rosbag_53", "rosbag_54", "rosbag_55",
            "rosbag_58", "rosbag_59", "rosbag_61", "rosbag_62", "rosbag_66", "rosbag_67", "rosbag_69",
            "rosbag_70", "rosbag_73", "rosbag_74", "rosbag_75", "rosbag_76", "rosbag_80", "rosbag_82",
            "rosbag_86", "rosbag_88", "rosbag_91", "rosbag_92", "rosbag_94", "rosbag_96", "rosbag_97", 
            "rosbag_102", "rosbag_105", "rosbag_107", "rosbag_108"]

if __name__ == "__main__":

# %% ## GRID TRAINER OPTIONS ## %% #
    # SELECT MODEL
    model = PINN()
    sim_model_name = "pinn"

    # SAVE NAME
    save_best_name = "pinn_best_grid.pt"
    save_model_name = "pinn.pt"

    # DIVISION FOR TRAIN / VALIDATE SPLIT
    rng_seed = 0
    train_procent = 0.8

    # HYPER - PARAMETERS
    n_steps = 20
    dropout_rate = 0.25

    # Use best perform here
    layer_grid = [25, 50]
    size_grid = [32, 64]
    factor_grid = [0.5, 0.25]
    lr0 = 0.001
    max_norm = 1.0
    patience = 5000
    epochs = 50000

    # INPUT - OUTPUT SHAPES
    input_shape = 19
    output_shape = 36 # 36 - 6x6 - D, 6 - nu_dot 

#####################################

# %% # 

    # Divide up the datasets into training and validation
    random.seed(rng_seed)
    random.shuffle(datasets)
    train_val_split = int(np.shape(datasets)[0] * train_procent)

    # Load training data
    x_trajectories, y_trajectories = load_to_trajectory(datasets[:train_val_split])

    # Get normalization constants
    all_x = torch.cat(x_trajectories, dim=0)
    x_mean = torch.mean(all_x, dim=0)
    x_std = torch.std(all_x, dim=0) + 1e-8 # Preventing division by 0

    # # Overwrite normalization just to turn it off in an easy way
    # x_mean = 0
    # x_std = 1

    # Load validation data
    x_trajectories_val, y_trajectories_val = load_to_trajectory(datasets[train_val_split:])

    # Load test data
    x_trajectories_test, y_trajectories_test = load_to_trajectory(test_datasets)

    # For results
    error_grid = np.zeros((len(layer_grid), len(size_grid), len(factor_grid)))
    best_error = float("inf")

    # For plotting loss at end of training
    best_loss_history = []
    best_val_loss_history = []

    model_counter = 0
    # Grid params
    for i, layers in enumerate(layer_grid): # Amount of layers
        for j, size in enumerate(size_grid): # Amount of neurons in each layer 
            for k, factor in enumerate(factor_grid):
                
                model_counter += 1
                print(f" Starting training on model {model_counter} / {len(layer_grid) * len(size_grid) * len(factor_grid)}")

                loss_history = []
                val_loss_history = []

                # NN shape
                shape = [size] * layers # Hidden layers
                shape.insert(0, input_shape) # Input layer
                shape.append(output_shape) # Output layer
                
                # Initialize model
                if isinstance(model, BPINN):
                    model.initialize(shape, dropout_rate)
                else:
                    model.initialize(shape)
                
                # Optimizer and other settings
                optimizer = torch.optim.Adam(model.parameters(), lr=lr0)

                # Adaptive learning rate
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=factor, patience=2500, threshold=10, min_lr=1e-8)

                # Early stopping with validation loss
                best_val_loss = float("inf")
                counter = 0
                best_model_state = None

                # Training loop
                for epoch in range(epochs):

                    model.train()
                    optimizer.zero_grad()

                    # Forward pass for each of our trajectories
                    loss_total = 0.0
                    for x_traj, y_traj in zip(x_trajectories, y_trajectories):
                        
                        # Get PI loss
                        x_traj_normed = (x_traj - x_mean) / x_std
                        loss = model.loss_function(model, x_traj_normed, y_traj, n_steps)
                        loss_total += loss

                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) # Clipping to help with stability
                    optimizer.step()

                    # Evaluate model on validation data
                    model.eval()
                    val_loss_total = 0.0
                    with torch.no_grad():
                        for x_traj_val, y_traj_val in zip(x_trajectories_val, y_trajectories_val):

                            # Get PI validation loss
                            x_traj_val_normed = (x_traj_val - x_mean) / x_std
                            val_loss = model.loss_function(model, x_traj_val_normed, y_traj_val, n_steps)
                            val_loss_total += val_loss
                    
                    # Step scheduler
                    scheduler.step(loss_total)
                    
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
                    if epoch % 100 == 0:
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
                            "x_std": x_std,
                            "dropout": dropout_rate}, 
                            "src/smarc_modelling/piml/models/"+save_model_name) 
                model.eval() # Just to doubly ensure that it is in eval mode

                total_error = 0.0
                for x_traj_test, y_traj_test in zip(x_trajectories_test, y_trajectories_test):

                    # Initial position for flipping
                    eta_for_error = y_traj_test["eta"]
                    eta_test_degs = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_for_error])

                    # Getting the state vector in such a way that we can use it in the simulator
                    states_test = [x_traj_test[0:7][:], x_traj_test[7:13][:], x_traj_test[13:19][:]]

                    try: 
                        # Running the SAM simulator to get predicted validation path
                        sam = SIM(sim_model_name, states_test, y_traj_test["t"], y_traj_test["u_cmd"], False)
                        results, _ = sam.run_sim()
                        results = torch.tensor(results).T
                        eta_model = results[:, 0:7]
                        nu_model = results[:, 7:13]

                        # Convert quat to angles
                        eta_model_degs = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_model])

                        # Calculated summed error
                        eta_mse = np.array((eta_model_degs - eta_test_degs)**2)
                        nu_mse = np.array((nu_model - y_traj_test["nu"])**2)
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
                            torch.save({"model_shape": shape, 
                                        "state_dict": model.state_dict(), 
                                        "dropout": dropout_rate,
                                        "x_mean": x_mean,
                                        "x_std": x_std},
                                        "src/smarc_modelling/piml/models/"+save_best_name)
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