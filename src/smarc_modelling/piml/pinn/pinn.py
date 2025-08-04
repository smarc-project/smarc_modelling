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

datasets = ["rosbag_16", "rosbag_15", "rosbag_14", "rosbag_13",
            "rosbag_12", "rosbag_1", "rosbag_2", "rosbag_3", 
            "rosbag_4", "rosbag_5", "rosbag_6", "rosbag_7", 
            "rosbag_8", "rosbag_9", "rosbag_10", "rosbag_11"]

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
        
        # Activation function
        self.activation = nn.Tanh()

        # Creating output layer
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # # Scaling layer to revert D to the right scale after normalizing
        # self.scale_layer = nn.Sequential(nn.Linear(layer_sizes[-2], 1), nn.Softplus())

    def forward(self, x):
        # Apply activation function to all layers except last
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Getting final prediction without bounds to get full range prediction
        A_flat = self.output_layer(x)
        A_mat = A_flat.view(-1, 6, 6)
        # A_mat = A_mat / torch.norm(A_mat, dim=(1, 2), keepdim=True).clamp(min=1e-6)

        # # Predict scale factor
        # scale = self.scale_layer(x).view(-1 ,1 ,1)

        # Compute D
        D = (A_mat @ A_mat.transpose(-2, -1)) # * scale
        return D


def loss_function(model, x_traj, y_traj, n_steps=10):
    """
    Custom loss function that implements the physics loss.
    """

    # Unpacking inputs
    nu = x_traj[:, 6:12]

    # Output values
    Dv_comp = y_traj["Dv_comp"]
    Mv_dot = y_traj["Mv_dot"]
    Cv = y_traj["Cv"]
    g_eta = y_traj["g_eta"]
    tau = y_traj["tau"]
    M = y_traj["M"]
    nu_dot = y_traj["acc"]
    
    N = len(tau) # Amount of datapoints

    # Getting predicted D
    D_pred = model(x_traj)

    # Calculate MSE physics loss using Fossen Dynamics Model
    # Physics loss predicted D matrix here should sum all the forces to 0
    physics_loss = (1/N) * torch.mean((Mv_dot + Cv + (torch.bmm(D_pred, nu.unsqueeze(2)).squeeze(2)) + g_eta - tau)**2)

    # # Data Loss
    # rhs = tau.unsqueeze(0) - Cv.unsqueeze(0) - g_eta.unsqueeze(0) - torch.bmm(D_pred, nu.unsqueeze(2)).squeeze(2)
    # rhs = rhs.squeeze()
    # nu_dot_model = torch.linalg.solve(M, rhs) # Solve for the acceleration
    # data_loss = (1/N) * torch.mean((nu_dot_model - nu_dot)**2)

    # Multi-step loss
    data_loss = (1/N) * multi_step_loss_function(model, x_traj, y_traj, n_steps)
    
    # Encourage high damping in roll and y directions by returning high values when corresponding damping terms are low
    damping_penalty = additional_damping_penalty(D_pred)

    # Scaling to ensure losses have approximately same scale
    alpha = 0.05
    beta = 0.5
    gamma = 0.001

    # print("\n")
    # print(f"PL: {physics_loss*alpha}")
    # print(f"DL: {data_loss*beta}")
    # print(f"DP: {damping_penalty*gamma}")
    # print("\n")

    # Final loss is just the sum
    return physics_loss*alpha + data_loss*beta + damping_penalty*gamma


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

def additional_damping_penalty(D_pred):
    # Get the elements for damping in y and for roll
    y_damp = D_pred[:, 1, 1]
    roll_damp = D_pred[:, 3, 3]

    # Check if the damping is below the wanted threshold
    threshold = 60.0
    loss_y = torch.mean(torch.relu(threshold - y_damp)**2)
    loss_roll = torch.mean(torch.relu(threshold - roll_damp)**2)

    return loss_y + loss_roll


def init_pinn_model(file_name: str):
    # For easy initialization of model in other files
    dict_path = "src/smarc_modelling/piml/models/" + file_name
    dict_file = torch.load(dict_path, weights_only=True)
    model = PINN(dict_file["model_shape"])
    model.load_state_dict(dict_file["state_dict"])
    x_mean = dict_file["x_mean"]
    x_std = dict_file["x_std"]
    model.eval()
    return model, x_mean, x_std


# def init_pinn_model_hybrid():
#     # For easy initialization of model in other files

#     # Linear model
#     dict_path = "src/smarc_modelling/piml/models/pinn/pinn_lin.pt"
#     dict_file = torch.load(dict_path, weights_only=True)
#     model_lin = PINN(dict_file["model_shape"])
#     model_lin.load_state_dict(dict_file["state_dict"])
#     model_lin.eval()

#     # Rotational model
#     dict_path = "src/smarc_modelling/piml/models/pinn/pinn_rot.pt"
#     dict_file = torch.load(dict_path, weights_only=True)
#     model_rot = PINN(dict_file["model_shape"])
#     model_rot.load_state_dict(dict_file["state_dict"])
#     model_rot.eval()
#     return [model_lin, model_rot]


def pinn_predict(model, eta, nu, u, norm):
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


# def pinn_predict_hybrid(models, eta, nu, u):
    
#     # Flatten input
#     eta = np.array(eta, dtype=np.float32).flatten()
#     nu = np.array(nu, dtype=np.float32).flatten()
#     u = np.array(u, dtype=np.float32).flatten()

#     # Make state vector
#     x = np.concatenate([eta, nu, u], axis=0)
#     x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

#     lin_model = models[0]
#     rot_model = models[1]

#     D_lin = lin_model(x).detach().numpy()
#     D_rot = rot_model(x).detach().numpy()

#     D = np.eye(6)
#     D[:3, :] = D_lin[0][:3, :]
#     D[3:, :] = D_rot[0][3:, :]

#     return D.squeeze()


if __name__ == "__main__":
    
    # Quick listing of commands when running file
    if "help" in sys.argv:
        print(f""" Available command line arguments: 
 save: Saves the model dict to file.
 plot: Plots the loss and lr per epoch.""")

    # Divide up the datasets into training and validation
    train_procent = 0.9 # How much goes to training, the rest goes to validation
    train_val_split = int(np.shape(datasets)[0] * train_procent)

    # Loading training data
    print(f" Loading training data...")
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

    x_trajectories = [(x_traj - x_mean) / x_std for x_traj in x_trajectories]

    # Loading validation data
    print(f" Loading validation data...")
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


    # Initialize model and optimizer
    shape = [19, 8, 8, 8, 8, 8, 8, 36] # Hidden layers
    model = PINN(shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping to help with stability

    # Adaptive learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9, patience=2500, threshold=10, min_lr=1e-8)

    # For plotting loss and lr
    if "plot" in sys.argv:
        loss_history = []
        val_loss_history = []
        lr_history = []

    # Early stopping using validation loss
    best_val_loss = float("inf")
    # How long we wait before stopping training
    patience = 7500
    counter = 0
    # Saving the best model before overfitting takes place
    best_model_state = None

    # For the multi-step integration loss
    n_steps = 25

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
            x_traj_normed = (x_traj - x_mean) / x_std
            loss = loss_function(model, x_traj_normed, y_traj, n_steps)
            loss_total += loss

        # Forward pass on training and loss computation
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
                x_traj_normed_val = (x_traj_val - x_mean) / x_std
                val_loss = loss_function(model, x_traj_normed_val, y_traj_val, n_steps)
                val_loss_total += val_loss

        # Saving loss and learning rate for plotting
        if "plot" in sys.argv:
            loss_history.append(loss_total.item())
            val_loss_history.append(val_loss_total.item())
            lr_history.append(optimizer.param_groups[0]['lr'])

        # Early stopping based on validation loss
        if val_loss_total.item() < best_val_loss:
            best_val_loss = val_loss_total.item()
            counter = 0 # Reset counter
            best_model_state = model.state_dict() # Saving the best model
        else:
            counter += 1

        if epoch % 500 == 0:
            print(f" Still training, epoch {epoch}, loss: {loss_total.item()}, lr: {optimizer.param_groups[0]['lr']}, \n validation loss: {val_loss_total.item()}, counter: {counter}")
        
        if counter >= patience:
            print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}")
            # Restore the best model
            model.load_state_dict(best_model_state)
            break

    print(f" Training done!")
    
    if "save" in sys.argv:
        # Saving the best model state from training (as per based on validation loss)
        torch.save({"model_shape": shape, 
                    "state_dict": model.state_dict(), 
                    "x_mean": x_mean,
                    "x_std": x_std},
                    "src/smarc_modelling/piml/models/pinn_trained.pt")
        print(f" Model weights saved to models/pinn_trained.pt")

    if "plot" in sys.argv:
        # Plotting the loss and lr
        plt.style.use('science')

        # Loss
        ax = plt.subplot(1, 1, 1)
        plt.plot(loss_history, linestyle="-", color="green", label="Training loss")
        ax.set_yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("log(Loss)")
        plt.plot(val_loss_history, linestyle="-", color="red", label="Validation loss")
        plt.legend()

        # # Learning rate
        # plt.subplot(2, 1, 2)
        # plt.plot(lr_history, linestyle="-", label="Learning Rate")
        # plt.xlabel("Epoch")
        # plt.ylabel("Learning Rate")
        # plt.ylim(0, 0.012)
        
        # Display
        plt.show()