# control
### NMPC class
The vector **x** contains the state vector and control vector concatenated. The order of the state vector is:


**x** = [x, y, z, q_w, q_x, q_y, q_z, u, v, w, p, q, r, x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]


The vector **u** Consists of the rate of change of the control inputs, i.e. **$\dot{u}$**. They are declared in the following order:

$\dot{u}$ = [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2].

**NOTE:** This is valid for the SAM casadi model, as the rest of the information in the README. Other models may have different orders!
### __init\_\_(casadi_model, Ts, N_horizon, update_solver_settings)

#### casadi_model: 
The casadi model to be used
#### Ts: 
The desired sampling interval for the controller
#### N_horizon: 
Can be found in the init. It is the length of the prediction horizon (and control horizon since they are equal).
#### update_solver_settings: 
If true, it regenerates and rebuilds the solver. Do this if changes have been made in the tuning or anything else related to the NMPC class.


### export_dynamics_model(casadi_model)
Method to augment the state vector with the control vector to make it work with Acados.
#### casadi_model: 
The casadi model to be used

### setup()
This method setup all the constraints and tuning parameters for the NMPC and should be run. Lastly, it rebuilds the NMPC if 
        
        update_solver_settings = True


This should be done every time any setting has been changed. The setup() method is divided into three parts:

#### Cost setup
Here can all weight matrices be adjusted for the stage costs and terminal cost. The weight matrices are:
- **Q_diag**: State and control magnitude weight matrix. Standard values are:

        Q_diag[ 0:3 ] = 10     # Position:         standard 10
        Q_diag[ 3:7 ] = 10     # Quaternion:       standard 10
        Q_diag[ 7:10] = 1      # linear velocity:  standard 1
        Q_diag[10:13] = 1      # Angular velocity: standard 1

        Control magnitude weight matrix - Costs set according to Bryson's rule
        Q_diag[13:15] = 1e-4            # VBS, LCG:      Standard: 1e-4
        Q_diag[ 15  ] = 5e1             # stern_angle:   Standard: 50
        Q_diag[ 16  ] = 5e1             # rudder_angle:  Standard: 50
        Q_diag[17:  ] = 1e-5            # RPM1 And RPM2: Standard: 1e-5

- **R_diag**: Controller rate weight matrix. Standard values are:

        R_diag[ :2] = 1e-3
        R_diag[2:4] = 1e1
        R_diag[4: ] = 1e-5
        R = np.diag(R_diag)*1e-3

Feel free to further tune this and divide the grouped tuning parameters. This is just a basic tune.

#### Constraint setup
Here, constraints can be set on the states, control magnitude and controller rate of change.
Currently, **only** the controller magnitudes are limited to:

- **LCG and VBS**: 0-100
- **Horizontal and vertical thrust vectoring**: $\pm 7 \degree$ (but in radians)
- **RPMs**: $\pm 600$ rpm

and the rate of of change for the VBS and LCG to:
- **LCG**: $\pm 50$
- **VBS**: $\pm 200$

The rate of change limits needs to be further tuned since they don't really corresponds to the real SAM, yet!

#### Solver setup 
Settings to the solver can be adjusted here. Settings such as which solver to be used, settings for nonlinear problems, tolerances etc.

#### x_error
Method to calculate the state deviations between the current actual state and current desired state. It uses quaternion error and is used in the setup.

#### Output
The OCP is solved by running
            
            self.ocp_solver.solve()

Then the system needs to be integrated once to get **u** and not $\dot{u}$. Example of usage is:

            self.simU = self.ocp_solver.get(0, "u")
            mpc_solution = self.integrator.simulate(x=x_current, u=self.simU)

The solution lies in the mpc_solution, where indeces [13:19] are the optimal control inputs in the following order:

    [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]


# acados_Trajectory_simulator
Reads in a trajectory from .csv file and simulates the tracking with the NMPC. It also plot the reference and the actual trajectory. Can be viewed as an example. Not used anymore, can be removed.


# NMPC_sim

Reads in trajectories  multiple .csv files. Calculates statistics on the tracking and plot the reference and actual trajectory. Can be viewed as an example. Not used anymore, can be removed.

# Usage in DiveController.py
More specifically in the DiveControllerMPC class. Some key variables and functions are explained here. Everything that is used by the NMPC, such as states and waypoints, are converted to NED from ENU.


### __init\_\_
**build:** Variable that declares if the OCP should be generated and built.  (=update_solver_settings) 

**ref_is_traj:** Variable that declares if the reference to be tracked is a trajectory or waypoint. If it is a trajectory, set:

        ref_is_traj = True

**_initialized:** Variable that declares if the current state have been acquired.

**_acados_status:** Dictionary that is used to print the status of the solver while running the NMPC. Originally, the Acados solver only returns values 0-6 but now it is possible to see what each value means. 


### update

**Nsim:** Length of the trajectory. Used for the sub-trajectory assignments to the NMPC when tracking trajectories. Also used to print the progress along a trajectory. 

Current state 


### get_init_state
Get the state message in the vector format presented under NMPC class. The states are converted from the ENU frame to the NED frame. Init class has a small perturbation in the thrusters in order to avoid numerical problems. NOTE: In case of numerical problems, a small perturbation can also be added to the velocity in x.

### get_current_state
Get the state message in the vector format presented under NMPC class. The states are converted from the ENU frame to the NED frame

### How to start
        ros2 launch sam_diving_controller waypoint_nmpc.launch robot_name:=sam_name