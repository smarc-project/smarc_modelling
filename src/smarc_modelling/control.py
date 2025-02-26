# Script for the acados NMPC model
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
import casadi as ca

class NMPC:
    def __init__(self, model):
        self.ocp = AcadosOcp()
        self.model = model  # Must be an acados ocp-model