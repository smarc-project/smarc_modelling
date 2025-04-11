import casadi as ca
import numpy as np
from smarc_modelling.piml.pinn.pinn import init_pinn_model, pinn_predict

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
        self.x = ca.MX.sym('x', self.n,1)
        self.u = ca.MX.sym('u', self.m,1)

        # Model properties
        # Declaration of parameters for the BlueROV2 heavy configuration
        self.g = 9.81       # Gravity acc  [kgm/s2]
        self.V = 0.0134     # Volume of rov [m3]    0.011 enl. 6.dof
        self.mass = 14.57       # Mass [kg] including DVL and Dropper            11.5 enl. 6-dof mod...
        self.Ix = 0.2818      # Inertia x-axis [kgm2] including DVL and Dropper 0.16  enl. 6-DoF modelling...
        self.Iy = 0.2450      # Inertia y-axis [kgm2] including DVL and Dropper 0.16
        self.Iz = 0.3852      # Inertia z-axis [kgm2] including DVL and Dropper 0.16
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
        self.Xa = 6.36     # Added mass x-axis [kg]            6.36 #Enl Open-Source benchmark....
        self.Ya = 7.12     # Added mass y-axis [kg]            7.12
        self.Za = 18.68    # Added mass z-axis [kg]            18.68
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

        eta = x[:int(self.n/2)] # Earth fixed pos and angles
        nu = x[int(self.n/2):] # Body fixed velocities
        u = tao # Controls

        # Model matrices
        J = self.create_J(eta) # Create transformation matrix
        M = self.create_M() # nxn inertia matrix including hydrodynamic added mass
        C = self.create_C(nu) # nxn nonlinear matrix with Coriolis, centrifugal and added mass terms
        D = self.create_D(eta, nu, u) # nxn nonlinear matrix dissipative terms 
        T = self.T # mxn Thruster transform matrix
        F = self.create_F(tao) # mx1 Force vector
        g = self.create_g(eta) # nx1 vector of restoring forces and moments
        # B = ca.MX.zeros(self.n, self.n) #nxm thruster characteristic input matrix 
        M_inv = np.linalg.inv(M) # Invert M
        self.T_inv = np.linalg.pinv(T)
        self.T_hat_inv=self.Create_Ardu_T() # Creates inverse matrix for thruster conversion
        
        # Nonlinear model
        detadt = ca.mtimes(J,nu)
        dnudt = ca.mtimes(M_inv,(ca.mtimes(T,F) - g - ca.mtimes(D,nu) - ca.mtimes(C,nu))) # Missing tether
        self.dxdt = ca.vertcat(detadt, dnudt)
        self.dxdt_sym = ca.Function('dxdt', [x, tao], [self.dxdt])

        return self.dxdt # xdot
    

    def create_J(self,eta):
        """
        J matrix
        :param state: virtual vector, (body-fixed linear and angular velocities)
        :type state: ca.MX
        """
        #get angles
        phi = eta[3] #roll
        theta = eta[4] #pitch
        psi = eta[5] #yaw

        #submatrices
        zeros = np.zeros((3, 3)) #zeros
        J1 = ca.MX.zeros(3,3) #inertia matrix
        J1[0,0] = ca.cos(psi) * ca.cos(theta)
        J1[0,1] = -ca.sin(psi) * ca.cos(phi) + ca.cos(psi) * ca.sin(theta) * ca.sin(phi)
        J1[0,2] = ca.sin(psi) * ca.sin(phi) + ca.cos(psi) * ca.cos(phi) * ca.sin(theta)
        J1[1,0] = ca.sin(psi) * ca.cos(theta)
        J1[1,1] = ca.cos(psi) * ca.cos(phi) + ca.sin(phi) * ca.sin(theta) * ca.sin(psi)
        J1[1,2] = -ca.cos(psi) * ca.sin(phi) + ca.sin(theta) * ca.sin(psi) * ca.cos(phi)
        J1[2,0] = -ca.sin(theta)
        J1[2,1] = ca.cos(theta) * ca.sin(phi)
        J1[2,2] = ca.cos(theta) * ca.cos(phi)

        J2 = ca.MX.zeros(3,3)
        J2[0,0] = 1
        J2[0,1] = ca.sin(phi)*ca.tan(theta)
        J2[0,2] = ca.cos(phi)*ca.tan(theta)
        J2[1,1] = ca.cos(phi)
        J2[1,2] = -ca.sin(phi)
        J2[2,1] = ca.sin(phi)/ca.cos(theta)
        J2[2,2] = ca.cos(phi)/ca.cos(theta)
        
        
        #build Rigid body inertia C_rb coriolis matrix
        row1 = ca.vertcat(J1,zeros)
        row2 = ca.vertcat(zeros,J2)
        J = ca.horzcat(row1,row2)
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
        I_matrix = ca.MX.zeros(3,3) #inertia matrix
        #print(I_matrix)
        I_matrix[0,1] = -self.Iz*r
        I_matrix[0,2] = -self.Iy*q
        I_matrix[1,0] = self.Iz*r
        I_matrix[1,2] = self.Ix*p
        I_matrix[2,0] = self.Iy*q
        I_matrix[2,1] = -self.Ix*p
        #print(I_matrix)

        #diag matrix
        diag_matrix = ca.MX.zeros(3,3)
        #print(diag_matrix)
        diag_matrix[0,1] = self.mass*w
        diag_matrix[0,2] = -self.mass*v
        diag_matrix[1,0] = -self.mass*w
        diag_matrix[1,2] = self.mass*u
        diag_matrix[2,0] = self.mass*v
        diag_matrix[2,1] = -self.mass*u
        #print(diag_matrix)
        
        #build Rigid body inertia C_rb coriolis matrix
        row1 = ca.vertcat(zeros,diag_matrix)
        row2 = ca.vertcat(diag_matrix,I_matrix)
        C_RB = ca.horzcat(row1,row2)
        #print(C_rb)

        #added mass coriolis
        matrix_c = ca.MX.zeros(3,3) #bottom right matrix
        #print(I_matrix)
        matrix_c[0,1] = -self.Na*r
        matrix_c[0,2] = self.Ma*q
        matrix_c[1,0] = self.Na*r
        matrix_c[1,2] = -self.Ka*p
        matrix_c[2,0] = -self.Ma*q
        matrix_c[2,1] = self.Ka*p
        #print(I_matrix)

        #diag matrix
        diag_matrix2 = ca.MX.zeros(3,3)
        #print(diag_matrix)
        diag_matrix2[0,1] = -self.Za*w
        diag_matrix2[0,2] = self.Ya*v
        diag_matrix2[1,0] = self.Za*w
        diag_matrix2[1,2] = -self.Xa*u
        diag_matrix2[2,0] = -self.Ya*v
        diag_matrix2[2,1] = self.Xa*u
        #print(diag_matrix)
        
        #build Rigid body inertia C_A coriolis matrix
        row1 = ca.vertcat(zeros,diag_matrix2)
        row2 = ca.vertcat(diag_matrix2,matrix_c)
        C_A = ca.horzcat(row1,row2)

        C =  C_RB + C_A
        return C
    

    def create_D(self, eta, nu, u):
        """
        D matrix
        """

        if self.piml_type == "":
            # White box matrix
            #get elements
            u = nu[0]
            v = nu[1]
            w = nu[2]
            p = nu[3]
            q = nu[4]
            r = nu[5]

            #linear damping
            D_l = ca.MX.eye(int(self.n/2))
            D_l[0,0] = self.Xul
            D_l[1,1] = self.Yvl
            D_l[2,2] = self.Zwl
            D_l[3,3] = self.Kpl
            D_l[4,4] = self.Mql
            D_l[5,5] = self.Nrl

            #nonlinear damping
            D_nl = ca.MX.eye(int(self.n/2))
            D_nl[0,0] = self.Xun*ca.fabs(u)
            D_nl[1,1] = self.Yvn*ca.fabs(v)
            D_nl[2,2] = self.Zwn*ca.fabs(w)
            D_nl[3,3] = self.Kpn*ca.fabs(p)
            D_nl[4,4] = self.Mqn*ca.fabs(q)
            D_nl[5,5] = self.Nrn*ca.fabs(r)

            D = D_l + D_nl

        if self.piml_type == "pinn":
            self.D = pinn_predict(self.piml_model, eta, nu, u)

        return D
    

    def create_F(self,tao):
        """
        returns symbolic vector of thruster force
        """

        F=-140.3*(tao**9)+389.9*(tao**7)-404.1*(tao**5)+176*(tao**3)+8.9*(tao**1)

        self.Force=ca.Function('F',[tao],[F])
        
        return F


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
        g = ca.MX.zeros(int(self.n/2),1)
        g[0] = (W-B)*ca.sin(theta)
        g[1] = -(W-B)*ca.cos(theta)*ca.sin(phi)
        g[2] = -(W-B)*ca.cos(theta)*ca.cos(phi)
        g[3] = self.yb*B*ca.cos(theta)*ca.cos(phi)-(self.zb*B*ca.cos(theta)*ca.sin(phi))
        g[4] = -self.zb*B*ca.sin(theta)-(self.xb*B*ca.cos(theta)*ca.cos(phi))
        g[5] = self.xb*B*ca.cos(theta)*ca.sin(phi)+(self.yb*B*ca.sin(theta))

        return g