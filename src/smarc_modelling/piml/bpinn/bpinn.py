#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports 
import torch
import torch.nn as nn
import numpy as np
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag
import sys
import matplotlib.pyplot as plt
import scienceplots # For fancy plotting

test_datasets = ["test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7", "test_8"]

datasets = ["rosbag_1", "rosbag_2", "rosbag_3", "rosbag_4", "rosbag_5", "rosbag_6", 
                  "rosbag_7", "rosbag_8", "rosbag_9", "rosbag_10", "rosbag_11", "rosbag_12"]

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

    # Final loss is just the sum
    loss = physics_loss + data_loss 

    return loss


def init_bpinn_model(file_name: str):
    # For easy initialization of model in other files
    dict_path = "src/smarc_modelling/piml/models/" + file_name
    dict_file = torch.load(dict_path, weights_only=True)
    model = BPINN(dict_file["model_shape"], dict_file["dropout"])
    model.load_state_dict(dict_file["state_dict"])
    model.eval()
    return model


def bpinn_predict(model, eta, nu, u):
    # For easy prediction in other files, this is the non-stochastic prediction

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
    
    # Divide up the datasets into training and validation
    train_procent = 0.8 # How much goes to training, the rest goes to validation
    train_val_split = int(np.shape(datasets)[0] * train_procent)

    # Loading training data
    print(f" Loading training data...")
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

    # Loading validation data
    print(f" Loading validation data...")
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

    # Initialize model and optimizer
    shape = [19, 32, 64, 128, 128, 64, 36]
    dropout = 0.1
    model = BPINN(shape, dropout)
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
    patience = 6000
    counter = 0
    # Saving the best model before overfitting takes place
    best_model_state = None

    # Training loop
    epochs = 500000
    print(f" Starting training...")
    for epoch in range(epochs):
        
        model.train()
        optimizer.zero_grad()

        # Forward pass for each trajectory
        loss_total = 0.0
        for x_traj, y_traj in zip(x_trajectories, y_trajectories):
            
            # Get the PI loss
            loss = loss_function(model, x_traj, y_traj["Dv_comp"], y_traj["Mv_dot"], y_traj["Cv"], y_traj["g_eta"], y_traj["tau"], y_traj["nu"])
            loss_total += loss
        loss_total.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step(loss_total)

        # Evaluate model on validation data
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x_traj_val, y_traj_val in zip(x_trajectories_val, y_trajectories_val):
               
                # Get PI validation loss
                val_loss = loss_function(model, x_traj_val, y_traj_val["Dv_comp"], y_traj_val["Mv_dot"], y_traj_val["Cv"], y_traj_val["g_eta"], y_traj_val["tau"], y_traj_val["nu"])
                val_loss_total += val_loss

        # Saving loss and learning rate for plotting
        if "plot" in sys.argv:
            loss_history.append(loss_total.item())
            val_loss_history.append(val_loss_total.item())
            lr_history.append(optimizer.param_groups[0]['lr'])

        # Early stopping based on validation loss
        if val_loss_total.item() < best_val_loss and epoch > 500:
            best_val_loss = val_loss.item()
            counter = 0 # Reset counter
            best_model_state = model.state_dict() # Saving the best model
        else:
            counter += 1

        if epoch % 500 == 0:
            print(f" Still training, epoch {epoch}, loss: {loss_total.item()}, lr: {optimizer.param_groups[0]['lr']}, \n validation loss: {val_loss_total.item()}")
        
        if counter >= patience:
            print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}")
            # Restore the best model
            model.load_state_dict(best_model_state)
            break

    print(f" Training done!")
    
    if "save" in sys.argv:
        torch.save({"model_shape": shape, "state_dict": model.state_dict(), "dropout": dropout}, "src/smarc_modelling/piml/models/bpinn.pt")
        print(f" Model weights saved to models/bpinn.pt")

    if "plot" in sys.argv:
        # Plotting the loss and lr
        plt.style.use('science')

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