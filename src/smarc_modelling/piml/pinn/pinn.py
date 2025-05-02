#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import torch
import torch.nn as nn
import numpy as np
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag
import sys
import matplotlib.pyplot as plt


class PINN(nn.Module):

    """
    Physics Informed Neural Network (PINN)
    Trains a damping matrix, D, that is strictly positive semi-definite, has
    positive diagonal elements and is symmetrical. This is done trough training
    using Cholesky decomposition.

    Also tries to uphold the Fossen Dynamics Equation trough physics loss function.
    """    

    def __init__(self, layer_sizes):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList()

        # Creating input layer
        self.layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))

        # Creating hidden layers
        for i in range(1, len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Creating output layer
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):
        # Apply activation function to all layers except last
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        # Getting final prediction without relu to get full range prediction
        A_flat = self.layers[-1](x)

        # Calculating D from A
        A_mat = A_flat.view(-1, 6, 6)
        D = A_mat @ A_mat.transpose(-2, -1)
        return D


def loss_function(model, x, Dv_comp, Mv_dot, Cv, g_eta, tau, nu):
    """
    Custom loss function that implements the physics loss.
    """
    
    # Getting predicted D
    D_pred = model(x)

    # Calculate MSE physics loss using Fossen Dynamics Model
    physics_loss = torch.mean((Mv_dot + Cv + (torch.bmm(nu.unsqueeze(1), D_pred).squeeze(1)) + g_eta - tau)**2)

    # Calculate data loss
    data_loss = torch.mean((Dv_comp - (torch.bmm(nu.unsqueeze(1), D_pred).squeeze(1)))**2)

    # Final loss is just the sum
    return physics_loss + data_loss


def init_pinn_model(file_name: str):
    # For easy initialization of model in other files
    dict_path = "src/smarc_modelling/piml/models/" + file_name
    dict_file = torch.load(dict_path, weights_only=True)
    model = PINN(dict_file["model_shape"])
    model.load_state_dict(dict_file["state_dict"])
    model.eval()
    return model


def pinn_predict(model, eta, nu, u):
    # For easy prediction in other files

    # Flatten input
    eta = np.array(eta, dtype=np.float32).flatten()
    nu = np.array(nu, dtype=np.float32).flatten()
    u = np.array(u, dtype=np.float32).flatten()

    # Make state vector
    x = np.concatenate([eta, nu, u], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    # Get prediction
    D_pred = model(x).detach().numpy()
    return D_pred.squeeze()


if __name__ == "__main__":
    
    # Quick listing of commands when running file
    if "help" in sys.argv:
        print(f""" Available command line arguments: 
 save: Saves the model dict to file.
 plot: Plots the loss and lr per epoch.""")
    
    # Loading training data
    print(f" Loading training data...")

    train_path = "src/smarc_modelling/piml/data/rosbags/rosbag_train"
    eta, nu, u, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t = load_data_from_bag(train_path, "torch")
    x = torch.cat([eta, nu, u], dim=1) # State vector

    # Loading validation data
    print(f" Loading validation data...")

    validate_path = "src/smarc_modelling/piml/data/rosbags/rosbag_validate"
    eta_val, nu_val, u_val, u_val_cmd, Dv_comp_val, Mv_dot_val, Cv_val, g_eta_val, tau_val, t_val = load_data_from_bag(validate_path, "torch")
    x_val = torch.cat([eta_val, nu_val, u_val], dim=1)

    # Initialize model and optimizer
    shape = [19, 32, 64, 128, 128, 64, 36]
    shape = [19, 256, 256, 256, 36]
    model = PINN(shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.95, patience=5000, threshold=1, min_lr=1e-5)

    # For plotting loss and lr
    if "plot" in sys.argv:
        loss_history = []
        val_loss_history = []
        lr_history = []

    # Early stopping using validation loss
    best_val_loss = float("inf")
    # How long we wait before stopping training
    patience = 5000
    counter = 0
    # Saving the best model before overfitting takes place
    best_model_state = None

    # Training loop
    epochs = 500000
    print(f" Starting training...")
    for epoch in range(epochs):
        
        model.train()
        optimizer.zero_grad()

        # Forward pass
        D_pred = model(x)

        # Forward pass on training and loss computation
        loss = loss_function(model, x, Dv_comp, Mv_dot, Cv, g_eta, tau, nu)
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step(loss)

        # Evaluate model on validation data
        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model, x_val, Dv_comp_val, Mv_dot_val, Cv_val, g_eta_val, tau_val, nu_val)

        # Saving loss and learning rate for plotting
        if "plot" in sys.argv:
            loss_history.append(loss.item())
            val_loss_history.append(val_loss.item())
            lr_history.append(optimizer.param_groups[0]['lr'])

        # Early stopping based on validation loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            counter = 0 # Reset counter
            best_model_state = model.state_dict() # Saving the best model
        else:
            counter += 1

        if epoch % 500 == 0:
            print(f" Still training, epoch {epoch}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
        
        if counter >= patience:
            print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}")
            # Restore the best model
            model.load_state_dict(best_model_state)
            break

    print(f" Training done!")
    
    if "save" in sys.argv:
        torch.save({"model_shape": shape, "state_dict": model.state_dict()}, "src/smarc_modelling/piml/models/pinn.pt")
        print(f" Model weights saved to models/pinn.pt")

    if "plot" in sys.argv:
        # Plotting the loss and lr

        # Loss
        ax = plt.subplot(2, 1, 1)
        plt.plot(loss_history, linestyle="-", color="green", label="Training loss")
        ax.set_yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("log(Loss)")
        plt.plot(val_loss_history, linestyle="-", color="red", label="Validation loss")
        plt.legend()

        # Learning rate
        plt.subplot(2, 1, 2)
        plt.plot(lr_history, linestyle="-", label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.ylim(0, 0.012)
        
        # Display
        plt.show()