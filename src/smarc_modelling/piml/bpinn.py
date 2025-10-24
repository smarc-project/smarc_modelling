#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports 
import torch
import torch.nn as nn
import numpy as np


# Functions and classes
class BPINN(nn.Module):

    """
    Bayesian Physics Informed Neural Network (B-PINN)
    Trains a damping matrix, D, that is strictly positive semi-definite, has
    positive diagonal elements and is symmetrical. This is done trough training
    using Cholesky decomposition.

    Also tries to uphold the Fossen Dynamics Equation trough physics loss function.

    Uses MC sampling and dropout during evaluation to get mean + standard deviation
    """    

    def __init__(self):
        super(BPINN, self).__init__()


    def initialize(self, layer_sizes, dropout_rate = 0.25):
        self.layers = nn.ModuleList()

        # Creating the input layer
        self.layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))

        # Creating hidden layers
        for i in range(1 , len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        # Creating the output layer 
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        # Activation function
        self.activation = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # Apply activation function and dropout to all layers except last
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        # Getting final prediction without relu to get full range prediction
        A_flat = self.layers[-1](x)

        # Calculate D from A
        A_mat = A_flat.view(-1, 6, 6)
        D = A_mat @ A_mat.transpose(-2, -1)
        return D
    

    def monte_carlo_forward(self, x, nu, num_samples):

        # Perform forward pass multiple times to gather samples
        preds = torch.stack([self.forward(x) for _ in range(num_samples)])  
        nu = torch.from_numpy(nu)

        # Compute the mean of D predictions
        D_pred_mean = preds.mean(dim=0) 

        # Perform matrix multiplication using torch.matmul for batch operation
        # We want the result to be [batch_size, 6, 1] after multiplication
        Dv_samples = torch.matmul(D_pred_mean, nu) 

        # Compute the mean and variance of D*v across the samples
        mean_Dv = Dv_samples.mean(dim=0)  # Mean of D*v across samples
        # var_Dv = Dv_samples.var(dim=0)  # Variance of D*v across samples
      
        return mean_Dv
    


    def loss_function(self, model, x_traj, y_traj, n_steps=10):
        """
        Custom loss function that implements the physics loss.
        """

        # Unpacking inputs
        nu = x_traj[:, 6:12]

        # Output values
        Mv_dot = y_traj["Mv_dot"]
        Cv = y_traj["Cv"]
        g_eta = y_traj["g_eta"]
        tau = y_traj["tau"]
        nu_dot = y_traj["acc"]
        M = y_traj["M"]
        
        # Getting predicted D
        D_pred = model(x_traj)

        # Calculate MSE physics loss using Fossen Dynamics Model
        # Physics loss predicted D matrix here should sum all the forces to 0
        physics_loss = torch.mean((Mv_dot + Cv + (torch.bmm(D_pred, nu.unsqueeze(2)).squeeze(2)) + g_eta - tau)**2)

        # # Multi-step loss
        # data_loss = multi_step_loss_function(model, x_traj, y_traj, n_steps)

        rhs = (tau - Cv - torch.matmul(D_pred, nu.unsqueeze(-1)).squeeze(-1) - g_eta)
        nu_dot_pred = torch.bmm(torch.linalg.inv(M), rhs.unsqueeze(-1)).squeeze(-1)
        data_loss = torch.mean((nu_dot_pred - nu_dot)**2)
        
        # Encourage high damping in roll and y directions by returning high values when corresponding damping terms are low
        damping_penalty = additional_damping_penalty(D_pred)

        # Scaling to ensure losses have approximately same scale
        alpha = 0.05
        beta = 0.5
        gamma = 0.001

        # Final loss is just the sum
        return physics_loss*alpha + data_loss*beta + damping_penalty*gamma


def multi_step_loss_function(model, x_traj, y_traj, h_steps):
    """
    Computes a multi-step integration loss based on nu over multiple steps
    """

    if h_steps == 0:
        h_steps = 1

    dt = torch.diff(y_traj["t"])
    loss = 0.0
    runs = 0

    eta = y_traj["eta"]
    nu = y_traj["nu"]
    u = y_traj["u"]
    Cv = y_traj["Cv"]
    g_eta = y_traj["g_eta"]
    tau = y_traj["tau"]
    M = y_traj["M"]

    N = len(eta)

    for start_index in range(N - 1):
        nu_pred = nu[start_index].unsqueeze(0) # x0
        run_loss = 0.0

        for step in range(h_steps):
            i = start_index + step
            if i >= N - 1:
                break

            # Pred input for model
            x_input = torch.cat([eta[i, 3:], nu_pred.squeeze(0), u[i, :]])
            D_pred = model(x_input)

            # Loss as difference between real velocity and collected for the next step
            run_loss += torch.mean((nu_pred - nu[i].unsqueeze(0))**2) 
            
            # Euler forward integration
            rhs = tau[i].unsqueeze(0) - Cv[i].unsqueeze(0) - g_eta[i].unsqueeze(0) - torch.bmm(D_pred, nu_pred.unsqueeze(2)).squeeze(2)
            rhs = rhs.squeeze()
            nu_dot = torch.linalg.solve(M[i], rhs) # Solve for the acceleration
            nu_pred = nu_pred + dt[i] * nu_dot # Euler forward with variable dt to get model difference to real value one step ahead

        # Loss
        runs += 1
        loss += run_loss / h_steps

    return loss / runs


def additional_damping_penalty(D_pred):
    # Get the elements for damping in y and for roll
    y_damp = D_pred[:, 1, 1]
    roll_damp = D_pred[:, 3, 3]

    # Check if the damping is below the wanted threshold
    threshold = 60.0
    loss_y = torch.mean(torch.relu(threshold - y_damp)**2)
    loss_roll = torch.mean(torch.relu(threshold - roll_damp)**2)

    return loss_y + loss_roll


def init_bpinn_model(file_name: str):
    # For easy initialization of model in other files
    dict_path = "src/smarc_modelling/piml/models/" + file_name
    dict_file = torch.load(dict_path, weights_only=True)
    model = BPINN()
    model.initialize(dict_file["model_shape"], dict_file["dropout"])
    model.load_state_dict(dict_file["state_dict"])
    x_min = dict_file["x_min"]
    x_range = dict_file["x_range"]
    model.eval()
    return model, x_min, x_range


def bpinn_predict(model, eta, nu, u, norm):
    # For easy prediction in other files, this is the non-stochastic prediction

    # Flatten input
    eta = np.array(eta, dtype=np.float32).flatten()
    nu = np.array(nu, dtype=np.float32).flatten()
    u = np.array(u, dtype=np.float32).flatten()

    # Make state vector
    x = np.concatenate([nu, u], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    x_normed = (x - norm[0]) / (norm[1] + 10e-8)

    # Get prediction, atm we do not use the std for anything
    model.train()
    with torch.no_grad():
        Dv_pred = model.monte_carlo_forward(x_normed, nu, 100)

    Dv_pred = np.array(Dv_pred)

    return Dv_pred.squeeze()
