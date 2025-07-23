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

# Datasets
datasets = [
    "rosbag_1", "rosbag_2", "rosbag_3", "rosbag_4",
    "rosbag_5", "rosbag_6", "rosbag_7", "rosbag_8",
    "rosbag_9", "rosbag_10", "rosbag_11", "rosbag_12",
    "rosbag_13", "rosbag_14", "rosbag_15", "rosbag_16"
]

datasets = ["rosbag_16", "rosbag_15", "rosbag_14", "rosbag_13", 
            "rosbag_12", "rosbag_1", "rosbag_2", "rosbag_3", 
            "rosbag_4", "rosbag_5", "rosbag_6", "rosbag_7", 
            "rosbag_8", "rosbag_9", "rosbag_10", "rosbag_11"]


class PINNN(nn.Module):
    """
    Physics-Informed Neural Network for AUV damping matrix estimation.
    """

    def __init__(self, layer_sizes):
        super().__init__()
        # Hidden and input layers
        self.layers = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        )
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.33)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            # x = self.dropout(x)
        A_flat = self.layers[-1](x)
        A_mat = A_flat.view(-1, 6, 6)
        D = A_mat @ A_mat.transpose(-2, -1)
        return D

def loss_function(model, x, y):
    """
    Combined physics loss and data loss
    """
    nu = x[:, 6:12]
    Mv_dot = y["Mv_dot"]
    Cv = y["Cv"]
    g_eta = y["g_eta"]
    tau = y["tau"]
    M = y["M"]
    acc = y["acc"]
    N = tau.size(0)

    D_pred = model(x)
    
    physics_loss = torch.mean((Mv_dot + Cv + torch.bmm(D_pred, nu.unsqueeze(2)).squeeze(2) + g_eta - tau)**2) / N
    
    rhs = tau - Cv - g_eta - torch.bmm(D_pred, nu.unsqueeze(2)).squeeze(2)
    acc_model = torch.linalg.solve(M, rhs)
    data_loss = torch.mean((acc_model - acc)**2)/N

    return physics_loss + data_loss

def init_pinn_model(file_name):
    """
    Initializes a PINN model from saved checkpoint.
    """
    dict_path = "src/smarc_modelling/piml/models" + file_name
    dict_file = torch.load(dict_path, weights_only=True)
    model = PINNN(dict_file["model_shape"])
    model.load_state_dict(dict_file["state_dict"])
    model.eval()
    return model

def pinn_predict(model, eta, nu, u):
    """
    Makes a single prediction with the given PINN model.
    """
    eta = np.array(eta, dtype=np.float32).flatten()
    nu = np.array(nu, dtype=np.float32).flatten()
    u = np.array(u, dtype=np.float32).flatten()
    x = np.concatenate([eta, nu, u], axis=0)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    D_pred = model(x).detach().numpy()
    return D_pred.squeeze()

if __name__ == "__main__":
    # Split datasets
    split_ratio = 0.9
    split = int(len(datasets) * split_ratio)
    train_ids = datasets[:split]
    val_ids = datasets[split:]

    # Load training data
    x_trajs, y_trajs = [], []
    for ds in train_ids:
        path =f"src/smarc_modelling/piml/data/rosbags/{ds}"
        eta, nu, u, _, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, acc = load_data_from_bag(path, "torch")
        x = torch.cat([eta, nu, u], dim=1)
        y = {"Dv_comp": Dv_comp, "Mv_dot": Mv_dot, "Cv": Cv,
             "g_eta": g_eta, "tau": tau, "M": M, "acc": acc}
        x_trajs.append(x); y_trajs.append(y)

    # Loading validation data
    x_vals, y_vals = [], []
    for ds in val_ids:
        path = f"src/smarc_modelling/piml/data/rosbags/{ds}"
        eta, nu, u, _, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, acc = load_data_from_bag(path, "torch")
        x = torch.cat([eta, nu, u], dim=1)
        y = {"Mv_dot": Mv_dot, "Cv": Cv,
             "g_eta": g_eta, "tau": tau, "M": M, "acc": acc}
        x_vals.append(x); y_vals.append(y)

    # Model, optimizer, scheduler
    shape = [19, 2, 2, 36]
    model = PINNN(shape)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Early stopping 
    counter = 0
    best_loss = float("inf")
    best_model_state = None
    patience = 1000

    # For plotting loss
    loss_history = []
    val_loss_history = []

    epochs = 50000
    for epoch in range(epochs):

        model.train()
        opt.zero_grad()

        # Forwards pass over all trajectories
        loss = 0.0
        for x_traj, y_traj in zip(x_trajs, y_trajs):
            loss += loss_function(model, x_traj, y_traj)

        loss.backward()
        opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in zip(x_vals, y_vals):
                val_loss += loss_function(model, x_val, y_val)

        # Saving current loss for plot
        if "plot" in sys.argv:
            loss_history.append(loss.item())
            val_loss_history.append(val_loss.item())

        # Early stopping by checking if val loss is going down
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1

        # Making sure the code is still running with update on current status
        if epoch % 500 == 0:
            print(f" Still training, epoch {epoch}, loss: {loss.item()}, lr: {opt.param_groups[0]['lr']}, \n validation loss: {val_loss.item()}, counter: {counter}")
        
        if counter >= patience:
            print(f" Stopping early due to no improvement after {patience} epochs from epoch: {epoch-counter}")
            # Restore the best model
            model.load_state_dict(best_model_state)
            break


    if "save" in sys.argv:
        # Saving the best model state from training (as per based on validation loss)
        torch.save({"model_shape": shape, 
                    "state_dict": model.state_dict(), 
                    },
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

        # Display
        plt.show()