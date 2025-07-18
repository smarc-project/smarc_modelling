#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smarc_modelling.piml.pinn.pinn import PINN, loss_function
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_deg
from smarc_modelling.piml.piml_sim import SIM
import torch
import numpy as np


if __name__ == "__main__":

    # Loading training data
    train_path = "src/smarc_modelling/piml/data/rosbags/rosbag_tank_1970"
    eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t = load_data_from_bag(train_path, "torch")
    x = torch.cat([eta, nu, u], dim=1) # State vector

    # Loading validation data
    validate_path = "src/smarc_modelling/piml/data/rosbags/rosbag_tank_2025"
    eta_val, nu_val, u_val, u_cmd_val, Dv_comp_val, Mv_dot_val, Cv_val, g_eta_val, tau_val, t_val = load_data_from_bag(validate_path, "torch")
    x_val = torch.cat([eta_val, nu_val, u_val], dim=1)

    # Loading the test data
    validate_path = "src/smarc_modelling/piml/data/rosbags/rosbag_tank_test"
    eta_test, nu_test, u_test, u_cmd_test, Dv_comp_test, Mv_dot_test, Cv_test, g_eta_test, tau_test, t_test = load_data_from_bag(validate_path, "torch")
    x_test = torch.cat([eta_test, nu_test, u_test], dim=1)

    # For flipping the sim results later
    x0 = eta_test[0, 0].item()
    y0 = eta_test[0, 1].item()
    z0 = eta_test[0, 2].item()
    eta_for_error = eta_test
    eta_for_error[:, 1] = 2 * y0 - eta_for_error[:, 1]
    eta_for_error[:, 2] = 2 * z0 - eta_for_error[:, 2]
    eta_test_degs = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_for_error])

    # For results
    error_grid = np.zeros((3, 3))
    best_error = float("inf")

    # Grid params
    for i, layers in enumerate([20, 50, 100]): # Amount of layers
        for j, size in enumerate([32, 64, 128]): # Amount of neurons in each layer 

                # NN shape
                shape = [size] * layers # Hidden layers
                shape.insert(0, 19) # Input layer
                shape.append(36) # Output layer
                
                # Initalize model
                model = PINN(shape)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                # Adaptive learning rate
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9, patience=500, threshold=0.01, min_lr=1e-5)

                # Early stopping with validation loss
                best_val_loss = float("inf")
                patience = 50000
                counter = 0
                best_model_state = None

                # Training loop
                epochs = patience * 10
                for epoch in range(epochs):

                    model.train()
                    optimizer.zero_grad()

                    # Forward pass
                    D_pred = model(x)

                    # Compute loss and optimize
                    loss = loss_function(model, x, Dv_comp, Mv_dot, Cv, g_eta, tau, nu)
                    loss.backward()
                    optimizer.step()

                    # Step scheduler
                    scheduler.step(loss)

                    # Evaluate model on validation data
                    model.eval()
                    with torch.no_grad():
                        val_loss = loss_function(model, x_val, Dv_comp_val, Mv_dot_val, Cv_val, g_eta_val, tau_val, nu_val)

                    # Handling for early stopping
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        counter = 0
                        best_model_state = model.state_dict()
                    else:
                        counter += 1

                    # Print statement to make sure everything is still running
                    if epoch % 500 == 0:
                        print(f" Still training, epoch {epoch}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']},\n validation loss: {val_loss.item()}, shape: {layers}, {size}")

                    # Break condition for early stopping
                    if counter >= patience:
                        print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}")
                        break
                
                # Calculating the model error
                model.load_state_dict(best_model_state) # Loading the best model state
                torch.save({"model_shape": shape, "state_dict": model.state_dict()}, "src/smarc_modelling/piml/models/pinn.pt") 
                model.eval() # Just to doubly ensure that it is in eval mode

                try: 
                    # Running the SAM simulator to get predicted validation path
                    sam_pinn = SIM("pinn", x_test[0], t_test, u_cmd_test)
                    results = sam_pinn.run_sim()
                    results = torch.tensor(results).T
                    eta_pinn = results[:, 0:7]
                    eta_pinn[:, 0] = 2 * x0 - eta_pinn[:, 0] # Flipping to NED frame
                    eta_pinn[:, 2] = 2 * z0 - eta_pinn[:, 2]
                    nu_pinn = results[:, 7:13]

                    # Convert quat to angles
                    eta_pinn_degs = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_pinn])

                    # Calculated summed error
                    eta_mse = np.array((eta_pinn_degs - eta_test_degs)**2)
                    nu_mse = np.array((nu_pinn - nu_test)**2)

                    error = np.sum(eta_mse) + np.sum(nu_mse)
                    error_grid[i,j] = error

                    if error < best_error:
                        best_error = error
                        torch.save({"model_shape": shape, "state_dict": model.state_dict()}, "src/smarc_modelling/piml/models/pinn_best_grid.pt")
                        best_setup = [layers, size]
    
                    print(" Succesful model made \n")

                except Exception as e:
                    # Since many of the models will be bad from the grid training,
                    # they will lead to the simulator going to inf and breaking it so we need to 
                    # have an except for these cases
                    print(f" {e}")
                    print(f" Error with simulator from faulty predictions\n")
                    error_grid[i,j] = float("inf")

                    print(f" Failed matrix was the following for example state {x_test[-50]}")
                    example = model(x_test[-50])
                    print(f" D: {example} \n")
                        

                
    # After going trough the grid getting the smallest loss
    min_loss_idx = np.unravel_index(np.argmin(error_grid), error_grid.shape)
    print(f" The best loss is at {min_loss_idx}")
    print(f" Best found configuration as: {best_setup}")

    # 6 - 128 best Jun 6 23:12
    # 12 - 32 best Jun 8 13:23
    # 20 - 32 best Jun 8 21:23