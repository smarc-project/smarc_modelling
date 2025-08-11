# control
### Containts the NMPC class

### Input (order)
- The vector **x** contains the state vector and control vector concatenated. The order of the state vector is:


        [x, y, z, q_w, q_x, q_y, q_z, u, v, w, p, q, r, x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]


- The vector **u** Consists of the rate of change of the control inputs, i.e. **u ̇_dot**. They are declared as following:


        [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
### Output (order)
Outputs the optimal control as [x_vbs_dot, x_lcg_dot, delta_s_dot, delta_r_dot, rpm1, rpm2]

#### Tuning
- N_horizon: Length of the prediction horizon
- Q_diag: State and control magnitude weight matrix
- R_diag: Controller rate weight matrix




# acados_Trajectory_simulator



# NMPC_sim

Can be removed - simulates a trajectory readed from a csv-file
