import torch
import numpy as np
from smarc_modelling.piml.pinn.pinn import init_pinn_model, pinn_predict
from smarc_modelling.piml.utils.utiity_functions import load_data_from_bag

# Least Squares for getting the "true" terms of the D matrix
# since the PINN methods only learns D*v

if __name__ == "__main__":

    # Init model
    model = init_pinn_model("/pinn.pt")

    # Load data
    validate_path = "src/smarc_modelling/piml/data/rosbags/rosbag_validate"
    eta_val, nu_val, u_val, u_val_cmd, Dv_comp_val, Mv_dot_val, Cv_val, g_eta_val, tau_val, t_val = load_data_from_bag(validate_path, "torch")
    x_val = torch.cat([eta_val, nu_val, u_val], dim=1)

    N = len(x_val)
    D_data = np.zeros((N, 6, 6))
    with torch.no_grad():
        for i in range(N):
            D = model(x_val[i])  # Should return (6, 6)
            D_data[i] = D.numpy()

    # Linear and quadratic parts
    L = np.zeros((6, 6))
    Q = np.zeros((6, 6))

    for i in range(6):  # row of D
        for j in range(6):  # column of D
            v = np.abs(nu_val[:, j])
            X = np.stack([np.ones(N), v], axis=1)  # (N, 2)
            y = D_data[:, i, j]                    # (N,)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            L[i, j], Q[i, j] = beta

    np.set_printoptions(precision=3, suppress=True)
    print("Linear damping matrix L:")
    print(L)
    print("Quadratic damping matrix Q:")
    print(Q)
    print("With following speed we get this example matrix:")
    print("Speed:\n", np.array(nu_val[-50, :]))
    D_example = L + np.diag(Q) * np.diag(np.abs(np.array(nu_val[-50, :])))
    print("D: \n", np.array(D_example))