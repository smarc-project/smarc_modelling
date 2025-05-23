#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from smarc_modelling.piml.pinn.pinn import PINN, loss_function
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag
from smarc_modelling.piml.piml_simulator import VEHICLE_SIM
import torch
import numpy as np


if __name__ == "__main__":

    # Loading training data
    train_path = "src/smarc_modelling/piml/data/rosbags/rosbag_tank_train"
    eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t = load_data_from_bag(train_path, "torch")
    x = torch.cat([eta, nu, u], dim=1) # State vector

    # Fixing coordinates for training data
    zeroing = eta[0, 0:3].clone()
    eta[:, 0:3] = eta[:, 0:3] - zeroing # Move data origin to 0
    eta[:, 0] = -eta[:, 0]
    eta[:, 1] = -eta[:, 1]
    eta[:, 2] = -eta[:, 2]
    eta[:, 0:3] = eta[:, 0:3] + zeroing

    # Loading validation data
    validate_path = "src/smarc_modelling/piml/data/rosbags/rosbag_tank_validate"
    eta_val, nu_val, u_val, u_cmd_val, Dv_comp_val, Mv_dot_val, Cv_val, g_eta_val, tau_val, t_val = load_data_from_bag(validate_path, "torch")
    x_val = torch.cat([eta_val, nu_val, u_val], dim=1)

    # Fixing coordinates for validation data
    zeroing = eta_val[0, 0:3].clone()
    eta_val[:, 0:3] = eta_val[:, 0:3] - zeroing # Move data origin to 0
    eta_val[:, 0] = -eta_val[:, 0]
    eta_val[:, 1] = -eta_val[:, 1]
    eta_val[:, 2] = -eta_val[:, 2]
    eta_val[:, 0:3] = eta_val[:, 0:3] + zeroing

    # For results
    loss_grid = np.zeros((5, 5, 5))
    best_error = float("inf")

    # Grid params
    for i, layers in enumerate([3, 4, 6, 8, 10]): # Amount of layers
        for j, size in enumerate([16, 32, 64, 128, 256]): # Amount of neurons in each layer 
            for k, lr_0 in enumerate([0.005, 0.01, 0.05, 0.1, 0.5]): # Starting learning rate <-- Changed by the scheduler during training later

                # NN shape
                shape = [size] * layers # Hidden layers
                shape.insert(0, 19) # Input layer
                shape.append(36) # Output layer
                
                # Initalize model
                model = PINN(shape)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)

                # Adaptive learning rate
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9, patience=500, threshold=0.01, min_lr=1e-5)

                # Early stopping with validation loss
                best_val_loss = float("inf")
                patience = 50000
                counter = 0
                best_model_state = None

                # Training loop
                epochs = 500000
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

                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        counter = 0
                        best_model_state = model.state_dict()
                    else:
                        counter += 1

                    if epoch % 500 == 0:
                        print(f" Still training, epoch {epoch}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']},\n validation loss: {val_loss.item()}, shape: {shape}")
                
                    if counter >= patience:
                        print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}")
                        break
                
                # Calculating the model error
                model.load_state_dict(best_model_state) # Loading the best model state
                torch.save({"model_shape": shape, "state_dict": model.state_dict()}, "src/smarc_modelling/piml/models/pinn.pt") 
                model.eval() # Just to doubly ensure that it is in eval mode

                try: 
                    # Running the SAM simulator to get predicted validation path
                    dt = 0.01
                    sam_pinn = VEHICLE_SIM("pinn", dt, x_val[0], t_val, u_cmd_val, "SAM")
                    results = torch.Tensor(sam_pinn.run_sim()[:, 0:13]).T

                    # Pulling out corresponding index of times in eval data
                    max_index_val = np.shape(results)
                    time_index = torch.clamp((t_val * (1/dt)).long(), min=0, max=max_index_val)
                    selected_index_results = [results[i, :] for i in time_index]

                    # Error from results
                    error = np.array(x_val[0:13, :]) - np.array(selected_index_results) 
                    error = np.sum(error**2)
                    loss_grid[i,j,k] = error

                    if error < best_error:
                        best_error = error
                        torch.save({"model_shape": shape, "state_dict": model.state_dict()}, "src/smarc_modelling/piml/models/pinn_best_grid.pt")
                    
                    print(" Succesful model made \n")

                except:
                    # Since many of the models will be bad from the grid training,
                    # they will lead to the simulator going to inf and breaking it so we need to 
                    # have an except for these cases
                    print(" Error with simulator from faulty predictions\n")
                    loss_grid[i,j,k] = float("inf")
                
    # After going trough the grid getting the smallest loss
    min_loss_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    print(f" The best loss is at {min_loss_idx}")