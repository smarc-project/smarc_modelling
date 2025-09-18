#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import torch
import torch.nn as nn
import numpy as np
from smarc_modelling.piml.utils.utility_functions import angular_vel_to_quat_vel, eta_quat_to_rad

class NaiveNN(nn.Module):

    """
    Naive neural network for comparison of performance with PINN. Does direct prediction of acceleration
    """    

    def __init__(self):
        super(NaiveNN, self).__init__()


    def initialize(self, layer_sizes):
        self.layers = nn.ModuleList()

        # Creating input layer
        self.layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))

        # Creating hidden layers
        for i in range(1, len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Activation function
        self.activation = nn.ReLU()

        # Creating output layer
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # Dropout for aiding with reduced overfitting
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        # Apply activation function to all layers except last
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        # Getting dynamics directly from prediction
        x_dot = self.output_layer(x)
        return x_dot


    def loss_function(self, model, x_traj, y_traj, n_steps=5):
        """
        Loss function
        """

        # # Get the two accelerations
        acc = y_traj["acc"]
        acc_pred = model(x_traj)  # (N, 6)
        
        # # Loss as difference
        loss = torch.mean((acc_pred - acc) ** 2)

        # loss = multi_step_loss_function(model, x_traj, y_traj, n_steps)

        return loss 
    
def multi_step_loss_function(model, x_traj, y_traj, h_steps):

    if h_steps == 0:
        h_steps = 1

    dt = torch.diff(y_traj["t"])
    loss = 0.0
    runs = 0
    
    nu = x_traj[:, :6]
    u = x_traj[:, 6:]
    nu_dot = y_traj["acc"]

    N = len(nu)

    for start_index in range(N - 1):
        nu_pred = nu[start_index]
        run_loss = 0.0

        for step in range(h_steps):
            i = start_index + step
            if i >= N - 1:
                break

            # Input vector
            x_input = torch.cat([nu_pred, u[i]])
            nu_dot_pred = model(x_input)

            # Loss
            run_loss += torch.mean((nu_dot_pred - nu_dot[i])**2)

            # EF integration for next step
            nu_pred = nu_pred + nu_dot_pred * dt[i]

        # Loss
        runs += 1
        loss += run_loss / h_steps

    return loss / runs


def init_naive_nn_model(file_name: str):
    # For easy initialization of model in other files
    dict_path = "src/smarc_modelling/piml/models/" + file_name
    dict_file = torch.load(dict_path, weights_only=True)
    model = NaiveNN()
    model.initialize(dict_file["model_shape"])
    model.load_state_dict(dict_file["state_dict"])
    x_min = dict_file["x_min"]
    x_range = dict_file["x_range"]
    y_min = dict_file["y_min"]
    y_range = dict_file["y_range"]
    model.eval()
    return model, x_min, x_range, y_min, y_range


def naive_nn_predict(model, eta, nu, u, norm):
    # For easy prediction in other files

    # norm = [x_min, x_range, y_min, y_range]

    # Flatten input
    eta = np.array(eta, dtype=np.float32).flatten()
    nu = np.array(nu, dtype=np.float32).flatten()
    u = np.array(u, dtype=np.float32).flatten()

    # Make state vector
    x = np.concatenate([nu, u], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    x_normed = (x - norm[0]) / (norm[1] + 10e-8)

    # Get prediction
    nu_dot = model(x_normed).detach().numpy()
    nu_dot =  nu_dot * norm[3].numpy() + norm[2].numpy()
    return nu_dot.squeeze()