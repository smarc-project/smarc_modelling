import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def linearize(u_nom, x_nom, f_disc, T):
    A_list, B_list = [], []

    A_sym = ca.jacobian(f_disc(x, u), x)
    B_sym = ca.jacobian(f_disc(x, u), u)
    A_func = ca.Function('A_func', [x, u], [A_sym])
    B_func = ca.Function('B_func', [x, u], [B_sym])

    for t in range(T):
        A_t = np.array(A_func(x_nom[t], u_nom[t]).full())
        B_t = np.array(B_func(x_nom[t], u_nom[t]).full())
        A_list.append(A_t)
        B_list.append(B_t)
    return A_list, B_list


def compute_gains(A_list, B_list, R, Q, T):
    P = Q.copy()
    K_list = [None] * (T - 1)

    for t in reversed(range(T - 1)):
        A = A_list[t]
        B = B_list[t]
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)  # K = (R + BᵀPB)⁻¹ BᵀPA
        K_list[t] = K
        P = Q + A.T @ P @ (A - B @ K)
    
    return K_list

# Time setup
T = 100   #len trajectory
dt = 0.1

# Define state and control
x = ca.MX.sym('x', 3)  # [x, y, theta]
u = ca.MX.sym('u', 2)  # [v, omega]

# Nonlinear dynamics: unicycle model
xdot = ca.vertcat(
    u[0] * ca.cos(x[2]),
    u[0] * ca.sin(x[2]),
    u[1]
)

# Create CasADi function
f = ca.Function('f', [x, u], [xdot])
f_disc = ca.Function('f_disc', [x, u], [x + dt * xdot])


x_nom = np.zeros((T, 3))
u_nom = np.zeros((T, 2))
x0 = np.array([0, 0, 0])

x_nom[0] = x0
for t in range(1, T):
    u_nom[t] = np.array([1.0, 0.1])  # constant velocity
    x_nom[t] = np.array(f_disc(x_nom[t-1], u_nom[t]).full()).flatten()


A_list, B_list = linearize(u_nom, x_nom, f_disc, T)


Q = np.diag([1.0, 1.0, 0.1])
R = np.diag([0.1, 0.1])

K_list = compute_gains(A_list, B_list, R, Q, T)


x_actual = np.zeros((T, 3))
x_actual[0] = x0


for t in range(T - 1):
    delta_x = x_actual[t] - x_nom[t]
    u_t = u_nom[t] - K_list[t] @ delta_x
    x_next = f_disc(x_actual[t], u_t).full().flatten()
    x_actual[t+1] = x_next



plt.figure(figsize=(8, 4))
plt.plot(x_nom[:, 0], x_nom[:, 1], 'r--', label="Nominal Trajectory")
plt.plot(x_actual[:, 0], x_actual[:, 1], 'b-', label="TV-LQR Controlled")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal')
plt.legend(); plt.grid(); plt.title("TV-LQR Tracking")
plt.show()
