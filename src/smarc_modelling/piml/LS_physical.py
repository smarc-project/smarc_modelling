import torch
import numpy as np
from smarc_modelling.piml.pinn.pinn import init_pinn_model, pinn_predict
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag

# Least Squares for getting the "true" terms of the D matrix
# since the PINN methods only learns D*v

if __name__ == "__main__":

    # Init model
    model = init_pinn_model("pinn.pt")

    # Load data
    validate_path = "src/smarc_modelling/piml/data/rosbags/rosbag_validate"
    eta_val, nu_val, u_val, u_val_cmd, Dv_comp_val, Mv_dot_val, Cv_val, g_eta_val, tau_val, t_val = load_data_from_bag(validate_path, "torch")
    x_val = torch.cat([eta_val, nu_val, u_val], dim=1)

    N = len(x_val)
    D_data = np.zeros((N, 6, 6))
    with torch.no_grad():
        for i in range(N):
            D = pinn_predict(model, eta_val[i, :], nu_val[i, :], u_val[i, :]) # model(x_val[i])  # (6, 6)
            D_data[i] = D

    # Getting Cholesky decomposition
    D_chol = np.zeros((N, 6, 6))
    for i in range(N):
        D = D_data[i]
        D_i = np.linalg.cholesky(D)
        D_chol[i] = D_i

    # Linear and quadratic parts
    L = np.zeros((6, 6))
    Q = np.zeros((6, 6))

    for i in range(6):
        for j in range(i+1): # Lower triangle form
            v = nu_val[:, j]
            X = np.stack([np.ones(len(v)), v], axis=1)
            y = D_chol[:, i, j]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            L[i, j], Q[i, j] = beta

    speed = np.array(nu_val[-50])
    D_chol_pred = np.zeros((6, 6))

    for i in range(6):
        for j in range(i+1):
            v = speed[j]
            D_chol_pred[i,j] = L[i,j] + Q[i,j]*v

    D_example = D_chol_pred @ D_chol_pred.T

    np.set_printoptions(precision=3, suppress=True)
    print("Linear damping matrix L:")
    print(L)
    print("Quadratic damping matrix Q:")
    print(Q)
    print("With following speed we get this example matrix:")
    print("Speed:\n", np.array(nu_val[-50, :]))
    print("D from LS: \n", np.array(D_example))
    pred_D = pinn_predict(model, eta_val[-50, :], nu_val[-50, :], u_val[-50, :])
    print("Predicted D: \n", pred_D)