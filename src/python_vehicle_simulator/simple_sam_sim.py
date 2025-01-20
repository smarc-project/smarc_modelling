import numpy as np
from scipy.integrate import solve_ivp
from python_vehicle_simulator.vehicles import *
from python_vehicle_simulator.lib import *
#from python_vehicle_simulator.vehicles.SAM import SAM
from python_vehicle_simulator.vehicles.SimpleSAM import SimpleSAM
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on what you have installed

# Initial conditions
eta0 = np.zeros(6)
#eta0[6] = 1.0  # Initial quaternion (no rotation) 
nu0 = np.zeros(6)  # Zero initial velocities
x0 = np.concatenate([eta0, nu0])

# Simulation timespan
t_span = (0, 10)  # 20 seconds simulation
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Create SAM instance
sam = SimpleSAM()

class Sol():

    def __init__(self, t, data) -> None:
        self.t = t
        self.y = data



def run_simulation(t_span, x0, sam):
    """
    Run SAM simulation using solve_ivp.
    """
    def dynamics_wrapper(t, x):
        """
        u: control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        """
        u = np.zeros(6)
        u[0] = 100
        return sam.dynamics(x, u)

    # Run integration
    print(f" Start simulation")

    data = np.empty((12, 500))
    data[:,0] = x0

    # Euler forward integration
    for i in range(500-1):
        data[:,i+1] = data[:,i] + dynamics_wrapper(0, data[:,i]) * (10/500)
    sol = Sol(t_eval,data)
    print(f" Simulation complete!")

    # RK 45 leads to numerical instabilities when setting the cb on top of the cg
#    sol = solve_ivp(
#        dynamics_wrapper,
#        t_span,
#        x0,
#        method='RK45',
#        t_eval=t_eval,
#        rtol=1e-6,
#        atol=1e-9
#    )
#    if sol.status == -1:
#        print(f" Simulation failed: {sol.message}")
#    else:
#        print(f" Simulation complete!")

    return sol


def plot_results(sol):
    """
    Plot simulation results.
    """

    def quaternion_to_euler_vec(sol):

        n = len(sol.y[3])
        psi = np.zeros(n)
        theta = np.zeros(n)
        phi = np.zeros(n)

        for i in range(n):
            q = [sol.y[3,i], sol.y[4,i], sol.y[5,i], sol.y[6,i]]
            psi[i], theta[i], phi[i] = gnc.quaternion_to_angles(q)

        return psi, theta, phi

    #psi_vec, theta_vec, phi_vec = quaternion_to_euler_vec(sol)
    psi_vec, theta_vec, phi_vec = sol.y[3,:], sol.y[4,:], sol.y[5,:] 

    fig, axs = plt.subplots(4, 3, figsize=(12, 10))

    # Position plots
    axs[0,0].plot(sol.t, sol.y[0], label='x')
    axs[0,1].plot(sol.t, sol.y[1], label='y')
    axs[0,2].plot(sol.t, -sol.y[2], label='z')
    axs[0,0].set_ylabel('x Position [m]')
    axs[0,1].set_ylabel('y Position [m]')
    axs[0,2].set_ylabel('-z Position [m]')
#    axs[0].legend()

    # Quaternion plots
#    axs[1,0].plot(sol.t, sol.y[3], label='q1')
#    axs[1,].plot(sol.t, sol.y[4], label='q2')
#    axs[1].plot(sol.t, sol.y[5], label='q3')
#    axs[1].plot(sol.t, sol.y[6], label='q0')
#    axs[1].set_ylabel('Quaternion')
#    axs[1].legend()

    # Euler plots
    axs[1,0].plot(sol.t, np.rad2deg(phi_vec), label='roll')
    axs[1,1].plot(sol.t, np.rad2deg(theta_vec), label='pitch')
    axs[1,2].plot(sol.t, np.rad2deg(psi_vec), label='yaw')
    axs[1,0].set_ylabel('roll [deg]')
    axs[1,1].set_ylabel('pitch [deg]')
    axs[1,2].set_ylabel('yaw [deg]')
#    axs[1].legend()

    # Velocity plots
    axs[2,0].plot(sol.t, sol.y[6], label='u')
    axs[2,1].plot(sol.t, sol.y[7], label='v')
    axs[2,2].plot(sol.t, sol.y[8], label='w')
    axs[2,0].set_ylabel('u (x_dot)')
    axs[2,1].set_ylabel('v (y_dot)')
    axs[2,2].set_ylabel('w (z_dot)')

    axs[3,0].plot(sol.t, sol.y[9], label='p')
    axs[3,1].plot(sol.t, sol.y[10], label='q')
    axs[3,2].plot(sol.t, sol.y[11], label='r')
    axs[3,0].set_ylabel('p (roll_dot)')
    axs[3,1].set_ylabel('q (pitch_dot)')
    axs[3,2].set_ylabel('r (yaw_dot)')
    #axs[3].legend()

#    # ksi plots
#    axs[4].plot(sol.t, sol.y[13], label='VBS')
#    axs[4].plot(sol.t, sol.y[14], label='LCG')
#    axs[4].plot(sol.t, sol.y[15], label='δs')
#    axs[4].plot(sol.t, sol.y[16], label='δr')
#    axs[4].plot(sol.t, sol.y[17], label='θ1')
#    axs[4].plot(sol.t, sol.y[18], label='θ2')
#    axs[4].set_ylabel('ksi')
#    axs[4].legend()
#
#    # ksi_dot plots
#    axs[5].plot(sol.t, sol.y[19], label='VBS')
#    axs[5].plot(sol.t, sol.y[20], label='LCG')
#    axs[5].plot(sol.t, sol.y[21], label='δs')
#    axs[5].plot(sol.t, sol.y[22], label='δr')
#    axs[5].plot(sol.t, sol.y[23], label='θ1')
#    axs[5].plot(sol.t, sol.y[24], label='θ2')
#    axs[5].set_ylabel('ksi_dot')
#    axs[5].legend()

    # ksi_ddot comparison
#    labels = ['VBS', 'LCG', 'δs', 'δr', 'θ1', 'θ2']
#    for i in range(6):
#        axs[6].plot(sol.t, ksi_ddot_unbounded[:, i], '--',
#                    label=f'{labels[i]} (unbounded)')
#        axs[6].plot(sol.t, ksi_ddot_bounded[:, i], '-',
#                    label=f'{labels[i]} (bounded)')
#    axs[6].set_ylabel('ksi_ddot')
#    axs[6].legend()

    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


# Run simulation and plot results
sol = run_simulation(t_span, x0, sam)
plot_results(sol)
