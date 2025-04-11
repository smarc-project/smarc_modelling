#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from smarc_modelling.piml.utils.utiity_functions import load_data_from_bag
import sys
import matplotlib.pyplot as plt

# Functions and classes
class BPINN(nn.Module):

    """
    Bayesian Physics Informed Neural Network (PINN)
    Trains a damping matrix, D, that is strictly positive semi-definite, has
    positive diagonal elements and is symmetrical. This is done trough training
    using Cholesky decomposition.

    Also tries to uphold the Fossen Dynamics Equation trough physics loss function.

    Uses MC sampling and dropout during evaluation to get mean + standard deviation
    """    

    def __init__(self, layer_sizes, dropout_rate = 0.1):
        super(BPINN, self).__init__()

        self.layers = nn.ModuleList()

        # Creating the input layer
        self.layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))

        # Creating hidden layers
        for i in range(1 , len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        # Creating the output layer 
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # Apply activation function and dropout to all layers except last
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        # Getting final prediction without relu to get full range prediction
        A_flat = self.layers[-1](x)

        # Calculate D from A
        A_mat = A_flat.view(-1, 6, 6)
        D = A_mat @ A_mat.transpose(-2, -1)
        return D
    

    def monte_carlo_forward(self, x, nu, num_samples):

        # Perform forward pass multiple times to gather samples
        preds = torch.stack([self.forward(x) for _ in range(num_samples)])  # Shape [num_samples, batch_size, 6, 6]
        
        # Compute the mean of D predictions
        D_pred_mean = preds.mean(dim=0)  # Shape [batch_size, 6, 6]
        
        # Ensure that nu has the shape [batch_size, 6, 1] for batch matrix multiplication
        nu_reshaped = nu.unsqueeze(2)  # Shape [batch_size, 6, 1]
        
        # Perform matrix multiplication using torch.matmul for batch operation
        # We want the result to be [batch_size, 6, 1] after multiplication
        Dv_samples = torch.matmul(D_pred_mean, nu_reshaped)  # Shape [batch_size, 6, 1]
        
        # Now squeeze the third dimension to get [batch_size, 6]
        Dv_samples = Dv_samples.squeeze(2)  # Shape [batch_size, 6]
        
        # Compute the mean and variance of D*v across the samples
        mean_Dv = Dv_samples.mean(dim=0)  # Mean of D*v across samples
        var_Dv = Dv_samples.var(dim=0)  # Variance of D*v across samples
        
        return mean_Dv, var_Dv
    

def loss_function(model, x, Dv_comp, Mv_dot, Cv, g_eta, tau, nu):
    """
    Computes the physics-informed loss by comparing NN output with expected calculations
    Sums this with the data loss (Dv_comp - Dv_pred)

    """
    
    # Getting the current prediction for D
    D_pred = model(x)
    
    # Calculate physics loss 
    # Enforce Fossen model
    physics_loss = torch.mean((Mv_dot + Cv + (torch.bmm(nu.unsqueeze(1), D_pred).squeeze(1)) + g_eta - tau)**2)

    # Calculate data loss
    data_loss = torch.mean((Dv_comp - (torch.bmm(nu.unsqueeze(1), D_pred).squeeze(1)))**2)

    # L1 norm to encourage sparsity / parsimony  <-- Actually this does not encourage sparsity or parsimony as it only affects the NN structure not the D matrix
    # NOTE: One way to actually do this would be summing all the elements in the matrix and presenting that as a loss
    l1_norm = 0 # sum(p.abs().sum() for p in model.parameters())

    loss = physics_loss + data_loss + l1_norm # We value learning the physics over just fitting the data

    return loss


def init_bpinn_model():
    # For easy initialization of model in other files
    dict_file = torch.load("src/smarc_modelling/piml/models/bpinn.pt", weights_only=True)
    model = BPINN(dict_file["model_shape"])
    model.load_state_dict(dict_file["state_dict"])
    model.eval()
    return model


def bpinn_predict(model, eta, nu, u):
    # For easy prediction in other files

    # Flatten input
    eta = np.array(eta, dtype=np.float32).flatten()
    nu = np.array(nu, dtype=np.float32).flatten()
    u = np.array(u, dtype=np.float32).flatten()

    # Make state vector
    x = np.concatenate([eta, nu, u], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    # Get prediction, atm this does not return the std
    D_pred = model(x).detach().numpy()
    return D_pred.squeeze()


def predict_with_uncertainty(model, x, nu, num_samples):
    model.train() # We stay in training mode to keep dropout enabled for sampling
    with torch.no_grad():
        mean_pred, uncertainty = model.monte_carlo_forward(x, nu, num_samples)
    return mean_pred, uncertainty


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
    model = BPINN(shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9, patience=500, threshold=0.01, min_lr=1e-5)

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
        torch.save({"model_shape": shape, "state_dict": model.state_dict()}, "src/smarc_modelling/piml/models/bpinn.pt")
        print(f" Model weights saved to models/bpinn.pt")

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