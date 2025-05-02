import numpy as np
from smarc_modelling.piml.pinn.pinn import init_pinn_model, pinn_predict
from smarc_modelling.piml.utils.utility_functions import eta_quat_to_deg, angular_vel_to_quat_vel
import os
import yaml

# Heavily modified to not use casadi anymore & implementing PIML stuff

class BlueROV_PIML(object):
    def __init__(self,
                 iface='casadi',
                 h=0.1,
                 tao = 0,
                 piml_type = None
                 ):
        """
        :param h: sampling time of the discrete system, defaults to 0.1
        :type h: float, optional
        :param tao: input time delay, defaults to 0
        :type tao: float, optional
        """
        # init
        self.solver = iface #solver, default is casadi
        self.n = 12 #no of states eta+nu combined
        self.m = 8 #no of inputs
        self.dt = h #sampling time

        # Casadi Variabels
        self.x = np.zeros((self.n, 1))
        self.u = np.zeros((self.m, 1))

        # Model properties
        # Declaration of parameters for the BlueROV2 heavy configuration
        # https://www.mdpi.com/2077-1312/10/12/1898
        self.g = 9.81         # Gravity acc  [kgm/s2]
        self.V = 0.0134       # Volume of rov [m3]    0.011 enl. 6.dof
        self.mass = 13.5     # Mass [kg] including DVL and Dropper             11.5  enl. 6-dof mod...
        self.Ix = 0.26      # Inertia x-axis [kgm2] including DVL and Dropper 0.16  enl. 6-DoF modelling...
        self.Iy = 0.23      # Inertia y-axis [kgm2] including DVL and Dropper 0.16
        self.Iz = 0.37      # Inertia z-axis [kgm2] including DVL and Dropper 0.16
        self.ro = 1000 #water density [kg/m3]
        self.delta = 0.0134  # water volume moved by ROV [m3] danish report

        # Center of bouyancy coordinates
        self.xb = 0         # Deviation from mass center in x-axis [m]
        self.yb = 0         # Deviation from mass center in y-axis [m]
        self.zb = -0.01     # Deviation from mass center in z-axis [m]

        # Center of gravity coordinates
        self.xg = 0         # x-axis [m]
        self.yg = 0         # y-axis [m]
        self.zg = 0         # z-axis [m]

        # Added masses
        self.Xa = 6.36      # Added mass x-axis [kg]            6.36 #Enl Open-Source benchmark....
        self.Ya = 7.12      # Added mass y-axis [kg]            7.12
        self.Za = 18.68     # Added mass z-axis [kg]            18.68
        self.Ka = 0.189     # Added inertia x-axis [kgm2/rad]   0.189
        self.Ma = 0.135     # Added inertia y-axis [kgm2/rad]   0.135
        self.Na = 0.222     # Added inertia z-axis [kgm2/rad]   0.222

        # Damping coefficients
        # - Linear
        self.Xul = 13.7     # Damping coefficient x-axis [Ns/m]
        self.Yvl = 0        # Damping coefficient y-axis [Ns/m]
        self.Zwl = 33       # Damping coefficient z-axis [Ns/m]
        self.Kpl = 0        # Damping coefficient rot. x-axis [Ns/m]
        self.Mql = 0.8      # Damping coefficient rot. y-axis [Ns/m]
        self.Nrl = 0        # Damping coefficient rot. z-axis [Ns/m]

        # - Nonlinear
        self.Xun = 141      # Damping coefficient x-axis [Ns2/m2]
        self.Yvn = 217      # Damping coefficient y-axis [Ns2/m2]
        self.Zwn = 190      # Damping coefficient z-axis [Ns2/m2]
        self.Kpn = 1.19     # Damping coefficient rot. x-axis [Ns2/m2]
        self.Mqn = 0.47     # Damping coefficient rot. y-axis [Ns2/m2]
        self.Nrn = 1.5      # Damping coefficient rot. z-axis [Ns2/m2]

        # Linearized model, continuous and discrete time
        self.Ac = None
        self.Bc = None
        self.C = None
        self.D = None
        self.Kc = None
        self.Ad = None
        self.Bd = None
        self.Kd = None
        self.B_delay = None #delay
        self.tao = tao #delay

        # Load yaml file
        config = self.load_file()

        # Get thruster configuration matrix T
        selected_T_config = config['selected_T_matrix']
        self.T_matrix_options = config['T_matrices']
        self.T = np.array(self.T_matrix_options[selected_T_config])

        # Get C and D matrices
        self.C = np.array(config['Model']['C_matrix'])
        self.D = np.array(config['Model']['D_matrix'])

        # Get no of references to track based on chosen C
        self.r = self.C.shape[0]

        # Physics informed machine learning approaches for D
        self.piml_type = piml_type
        if self.piml_type == "pinn":
            print(f" Physics Informed Neural Network model initialized!")
            self.piml_model = init_pinn_model("pinn_brov.pt")


    def dynamics(self, x, tao):
        """
        Nonlinear dynamics, without time delay

        :param x: state, (earth-fixed positions and euler angles) and virtual vector, (body-fixed linear and angular velocities) 
        :type x: ca.MX
        :param tao: control input, (body-fixed forces and moments), now changed to thruster ESC pwm signals
        :type tao: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """
        
        eta = eta_quat_to_deg(x[0:7])
        nu = x[7:13]
        u_fb = x[13:19]
        u = tao # Controls commands
    

        # Model matrices
        J = self.create_J(eta) # Create transformation matrix
        M = self.create_M() # nxn inertia matrix including hydrodynamic added mass
        C = self.create_C(nu) # nxn nonlinear matrix with Coriolis, centrifugal and added mass terms
        D = self.create_D(eta, nu, u) # nxn nonlinear matrix dissipative terms 
        FT = self.create_F(tao) # mx1 Force vector (already multiplied with T upon return)
        g = self.create_g(eta) # nx1 vector of restoring forces and moments
        M_inv = np.linalg.inv(M) # Invert M
  
        # Nonlinear model
        detadt = np.dot(J,nu) # This is in angles
        detadt = angular_vel_to_quat_vel(eta, detadt) # Convert it quaternion
        dnudt = (M_inv @ (FT.T - g.T - np.matmul(nu, D) - np.matmul(nu, C)).T).reshape(-1)
        dudt = [0, 0, 0, 0, 0, 0, 0, 0] # No control dynamics we read these from the bag
        self.dxdt = np.hstack((detadt, dnudt, dudt))
        
        return self.dxdt # xdot
    

    def create_J(self,eta):
        """
        J matrix
        :param state: virtual vector, (body-fixed linear and angular velocities)
        :type state: ca.MX
        """
        #get angles
        phi = eta[3] #roll (x)
        theta = eta[4] #pitch (y)
        psi = eta[5] #yaw (z)

        #submatrices
        zeros = np.zeros((3, 3)) #zeros
        J1 = np.zeros((3,3)) #inertia matrix
        J1[0,0] = np.cos(psi) * np.cos(theta)
        J1[0,1] = -np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(theta) * np.sin(phi)
        J1[0,2] = np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi) * np.sin(theta)
        J1[1,0] = np.sin(psi) * np.cos(theta)
        J1[1,1] = np.cos(psi) * np.cos(phi) + np.sin(phi) * np.sin(theta) * np.sin(psi)
        J1[1,2] = -np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi) * np.cos(phi)
        J1[2,0] = -np.sin(theta)
        J1[2,1] = np.cos(theta) * np.sin(phi)
        J1[2,2] = np.cos(theta) * np.cos(phi)

        J2 = np.zeros((3,3))
        J2[0,0] = 1
        J2[0,1] = np.sin(phi)*np.tan(theta)
        J2[0,2] = np.cos(phi)*np.tan(theta)
        J2[1,1] = np.cos(phi)
        J2[1,2] = -np.sin(phi)
        J2[2,1] = np.sin(phi)/np.cos(theta)
        J2[2,2] = np.cos(phi)/np.cos(theta)
        
        
        #build Rigid body inertia C_rb coriolis matrix
        row1 = np.vstack((J1,zeros))
        row2 = np.vstack((zeros,J2))
        J = np.hstack((row1,row2))
        return J


    def create_M(self):
        """
        M matrix
        return numpy array
        """

        # Rigid-body intertia
        m_matrix = np.eye(3)*self.mass #mass matrix
        I_matrix = np.diag([self.Ix, self.Iy, self.Iz])   # Inertia matrix
        zeros = np.zeros((3, 3)) #zeros
        M_RB = np.block([[m_matrix, zeros], [zeros, I_matrix]]) # Rigid-body inertia matrix
        #print(M_RB)
        # # Added mass inertia neglect offdiagonal common assumption, assume inviscid fluid and not at seabed and at surface
        M_A  = np.diag([self.Xa, self.Ya, self.Za, self.Ka, self.Ma, self.Na])
        #print(M_A)

        # Inertia matrix
        M = M_RB + M_A
        return M


    def create_C(self,nu):
        """
        C matrix
        :param state: virtual vector, (body-fixed linear and angular velocities)
        :type state: ca.MX
        """
        #get elements
        u = nu[0]
        v = nu[1]
        w = nu[2]
        p = nu[3]
        q = nu[4]
        r = nu[5]

        #submatrices
        zeros = np.zeros((3, 3)) #zeros
        I_matrix = np.zeros((3,3)) #inertia matrix
        #print(I_matrix)
        I_matrix[0,1] = -self.Iz*r
        I_matrix[0,2] = -self.Iy*q
        I_matrix[1,0] = self.Iz*r
        I_matrix[1,2] = self.Ix*p
        I_matrix[2,0] = self.Iy*q
        I_matrix[2,1] = -self.Ix*p
        #print(I_matrix)

        #diag matrix
        diag_matrix = np.zeros((3,3))
        #print(diag_matrix)
        diag_matrix[0,1] = self.mass*w
        diag_matrix[0,2] = -self.mass*v
        diag_matrix[1,0] = -self.mass*w
        diag_matrix[1,2] = self.mass*u
        diag_matrix[2,0] = self.mass*v
        diag_matrix[2,1] = -self.mass*u
        #print(diag_matrix)
        
        #build Rigid body inertia C_rb coriolis matrix
        row1 = np.vstack((zeros,diag_matrix))
        row2 = np.vstack((diag_matrix,I_matrix))
        C_RB = np.hstack((row1,row2))
        #print(C_rb)

        #added mass coriolis
        matrix_c = np.zeros((3,3)) #bottom right matrix
        #print(I_matrix)
        matrix_c[0,1] = -self.Na*r
        matrix_c[0,2] = self.Ma*q
        matrix_c[1,0] = self.Na*r
        matrix_c[1,2] = -self.Ka*p
        matrix_c[2,0] = -self.Ma*q
        matrix_c[2,1] = self.Ka*p
        #print(I_matrix)

        #diag matrix
        diag_matrix2 = np.zeros((3,3))
        #print(diag_matrix)
        diag_matrix2[0,1] = -self.Za*w
        diag_matrix2[0,2] = self.Ya*v
        diag_matrix2[1,0] = self.Za*w
        diag_matrix2[1,2] = -self.Xa*u
        diag_matrix2[2,0] = -self.Ya*v
        diag_matrix2[2,1] = self.Xa*u
        #print(diag_matrix)
        
        #build Rigid body inertia C_A coriolis matrix
        row1 = np.vstack((zeros,diag_matrix2))
        row2 = np.vstack((diag_matrix2,matrix_c))
        C_A = np.hstack((row1,row2))

        C =  C_RB + C_A
        return C
    

    def create_D(self, eta, nu, u_):
        """
        D matrix
        """

        if self.piml_type == None:
            # White box matrix
            #get elements
            u = nu[0]
            v = nu[1]
            w = nu[2]
            p = nu[3]
            q = nu[4]
            r = nu[5]

            #linear damping
            D_l = np.eye(int(self.n/2))
            D_l[0,0] = self.Xul
            D_l[1,1] = self.Yvl
            D_l[2,2] = self.Zwl
            D_l[3,3] = self.Kpl
            D_l[4,4] = self.Mql
            D_l[5,5] = self.Nrl

            #nonlinear damping
            D_nl = np.eye(int(self.n/2))
            D_nl[0,0] = self.Xun*abs(u)
            D_nl[1,1] = self.Yvn*abs(v)
            D_nl[2,2] = self.Zwn*abs(w)
            D_nl[3,3] = self.Kpn*abs(p)
            D_nl[4,4] = self.Mqn*abs(q)
            D_nl[5,5] = self.Nrn*abs(r)

            D = D_l + D_nl

        if self.piml_type == "pinn":
            self.D = pinn_predict(self.piml_model, eta, nu, u_)

        return D
    

    def create_F(self, tao):
        """
        enables predicting force using np ndarrays
        """
        tao = tao.reshape((8, 1)) # Add dim for mult
        K = (6136*tao + 108700)/(tao**3 + 89*tao**2 + 9258*tao + 108700)
        K_mat = np.diag(K.flatten())
        tao = tao - 1500
        tao = np.matmul(K_mat, tao)

        # Define the nonlinear force expression
        F = -140.3 * (tao**9) + 389.9 * (tao**7) - 404.1 * (tao**5) + 176 * (tao**3) + 8.9 * tao

        # Return the force
        return np.matmul(self.T, F)


    def create_g(self,eta):
        """
        g vector (restoring forces)
        """
        #get angles
        phi = eta[3] #roll
        theta = eta[4] #pitch

        #define forces
        W = self.mass*self.g #weight
        B = self.ro*self.g*self.delta #bouyancy

        #populate g vector
        g = np.zeros((int(self.n/2),1))
        g[0] = (W-B)*np.sin(theta)
        g[1] = -(W-B)*np.cos(theta)*np.sin(phi)
        g[2] = -(W-B)*np.cos(theta)*np.cos(phi)
        g[3] = self.yb*B*np.cos(theta)*np.cos(phi)-(self.zb*B*np.cos(theta)*np.sin(phi))
        g[4] = -self.zb*B*np.sin(theta)-(self.xb*B*np.cos(theta)*np.cos(phi))
        g[5] = self.xb*B*np.cos(theta)*np.sin(phi)+(self.yb*B*np.sin(theta))

        return g
    
    def load_file(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir = os.path.join(script_dir, '..', 'piml', 'brov', 'config')
        mpc_param_file = config_dir+"/MPC_Params.yaml"
        # print("script: "+ mpc_param_file)
        # print(os.path.isfile(mpc_param_file))
        with open(mpc_param_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    # brov_mpc/brov_mpc/lib
    # brov_mpc/config