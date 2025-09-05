#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import torch
import torch.nn as nn
import numpy as np

class NN(nn.Module):

    """
    Standard neural network for comparison of performance with PINN
    """    

    def __init__(self):
        super(NN, self).__init__()


    def initialize(self, layer_sizes):
        self.layers = nn.ModuleList()

        # Creating input layer
        self.layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))

        # Creating hidden layers
        for i in range(1, len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Activation function
        self.activation = nn.Tanh()

        # Creating output layer
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # Dropout for aiding with reduced overfitting
        self.dropout = nn.Dropout(p=0.25)


    def forward(self, x):
        # Apply activation function to all layers except last
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        # Getting D directly without ensuring symmetry with cholesky decomposition
        D_flat = self.output_layer(x)
        D = D_flat.view(-1, 6, 6)
        return D


    def loss_function(self, model, x_traj, y_traj, n_steps=10):
        """
        Custom loss function that implements the physics loss.
        """
        tau = y_traj["tau"]
        N = len(tau) # Amount of datapoints

        # Multi-step loss
        data_loss = (1/N) * multi_step_loss_function(model, x_traj, y_traj, n_steps)
        
        # Final loss is just the sum
        return data_loss


def multi_step_loss_function(model, x_traj, y_traj, n_steps=10):
    """
    Computes a multi-step integration loss based on nu over multiple steps
    """

    if n_steps == 0:
        print(f" n_steps was set to 0, changing it to 1!")
        n_steps = 1

    # Unpacking inputs
    eta = x_traj[:, :6]
    nu = x_traj[:, 6:12]
    u = x_traj[:, 12:]

    # Output values
    Cv = y_traj["Cv"]
    g_eta = y_traj["g_eta"]
    tau = y_traj["tau"]
    t = y_traj["t"]
    M = y_traj["M"]

    # Get the dt vector
    dt_np = np.diff(t.numpy())
    dt = torch.tensor(dt_np, dtype=torch.float32)
    
    loss = 0.0
    nu_pred = nu[0].unsqueeze(0) # x0

    for i in range(n_steps):
        if i >= len(eta) - 1:
            # Making sure we have values to compare to
            break

        # Prep inputs for model
        x_input = torch.cat([eta[i].unsqueeze(0), nu_pred, u[i].unsqueeze(0)], dim=1)
        D_pred = model(x_input) 

        # Euler forward integration
        rhs = tau[i].unsqueeze(0) - Cv[i].unsqueeze(0) - g_eta[i].unsqueeze(0) - torch.bmm(D_pred, nu_pred.unsqueeze(2)).squeeze(2)
        rhs = rhs.squeeze()
        nu_dot = torch.linalg.solve(M[i], rhs) # Solve for the acceleration
        nu_pred = nu_pred + dt[i] * nu_dot # Euler forward with variable dt to get model difference to real value one step ahead

        # Loss as difference between real velocity and collected for the next step
        loss += torch.mean((nu_pred.clone() - nu[i+1].unsqueeze(0))**2)

    return loss / n_steps


def init_nn_model(file_name: str):
    # For easy initialization of model in other files
    dict_path = "src/smarc_modelling/piml/models/" + file_name
    dict_file = torch.load(dict_path, weights_only=True)
    model = NN()
    model.initialize(dict_file["model_shape"])
    model.load_state_dict(dict_file["state_dict"])
    x_mean = dict_file["x_mean"]
    x_std = dict_file["x_std"]
    model.eval()
    return model, x_mean, x_std


def nn_predict(model, eta, nu, u, norm):
    # For easy prediction in other files

    # Flatten input
    eta = np.array(eta, dtype=np.float32).flatten()
    nu = np.array(nu, dtype=np.float32).flatten()
    u = np.array(u, dtype=np.float32).flatten()

    # Make state vector
    x = np.concatenate([eta, nu, u], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    x_normed = (x - norm[0]) / norm[1]

    # Get prediction
    D_pred = model(x_normed).detach().numpy()
    return D_pred.squeeze()