import numpy as np
import control as ctrl
import sympy as sp
from sympy import sin, cos, tan

class DroneModel:
    def __init__(self, Ts):
        
        self.m = 0.65
        self.Ix = 7.5e-3
        self.Iy = 7.5e-3
        self.Iz = 1.3e-2
        self.Ts = Ts
        self.T = 0

        
        x, y, z, u, v, w = sp.symbols('x y z u v w')
        phi, theta, psi = sp.symbols('phi theta psi')
        p, q, r = sp.symbols('p q r')
        u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')
        m, g = sp.symbols('m g')
        Ix, Iy, Iz = sp.symbols('I_x I_y I_z')

        state_syms = sp.Matrix([x, y, z, u, v, w, phi, theta, psi, p, q, r])
        input_syms = sp.Matrix([u1, u2, u3, u4])

        # Nonlinear dynamics
        dx = u*(cos(theta)*cos(psi)) + v*(sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)) + w*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
        dy = u*(cos(theta)*sin(psi)) + v*(sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi)) + w*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
        dz = u*(sin(theta)) - v*(sin(phi)*cos(theta)) - w*(cos(phi)*cos(theta))
        du = r*v - q*w - g*sin(theta)
        dv = p*w - r*u + g*cos(theta)*sin(phi)
        dw = q*u - p*v + g*cos(theta)*cos(phi) - u1/m
        dphi = p + q*(sin(phi)*tan(theta)) + r*(cos(phi)*tan(theta))
        dtheta = q*cos(phi) - r*sin(phi)
        dpsi = (q*sin(phi) + r*cos(phi))/cos(theta)
        dp = (Iy-Iz)/Ix * q*r + u2/Ix
        dq = (Iz-Ix)/Iy * p*r + u3/Iy
        dr = (Ix-Iy)/Iz * p*q + u4/Iz

        f = sp.Matrix([dx, dy, dz, du, dv, dw, dphi, dtheta, dpsi, dp, dq, dr])
        Asym = f.jacobian(state_syms)
        Bsym = f.jacobian(input_syms)

        # Lambdified functions 
        self.A_func = sp.lambdify((x, y, z, u, v, w, phi, theta, psi, p, q, r,
                                   u1, u2, u3, u4, m, g, Ix, Iy, Iz), Asym, 'numpy')
        self.B_func = sp.lambdify((x, y, z, u, v, w, phi, theta, psi, p, q, r,
                                   u1, u2, u3, u4, m, g, Ix, Iy, Iz), Bsym, 'numpy')

    def computeA(self, current_state):
        # hover input as default trim
        u1_eq = self.m * 9.81
        u2_eq, u3_eq, u4_eq = 0, 0, 0
        #clipped_state = current_state.copy()
        #clipped_state[7] = np.clip(clipped_state[7], -np.pi/2 + 1e-3, np.pi/2 - 1e-3)

        values = list(current_state) + [u1_eq, u2_eq, u3_eq, u4_eq,
                                        self.m, 9.81, self.Ix, self.Iy, self.Iz]

        self.A = np.array(self.A_func(*values), dtype=np.float64)
        self.B = np.array(self.B_func(*values), dtype=np.float64)

        # Discretisation
        sys = ctrl.ss(self.A, self.B, np.eye(12), np.zeros((12, 4)))
        sysd = ctrl.c2d(sys, Ts=self.Ts)
        self.Ad = sysd.A
        self.Bd = sysd.B


    def _dynamics(self, state_vector: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        state_vector = state_vector.flatten('C')
        control_input = control_input.flatten('C')
        return self.A @ state_vector + self.B @ control_input
    
    def _wind_disturbance(self, T: float) -> np.ndarray:
        disturbance = np.zeros((12,1))
        disturbance[4] = np.sin(0.1*T) + np.random.normal(0,0.1)
        disturbance[5] = np.cos(0.1*T) + np.random.normal(0,0.1)
        return disturbance.flatten('C')
    
    def discrete_dynamics(self, state_vector: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        state_vector = state_vector.flatten('C')
        control_input = control_input.flatten('C')
        return self.Ad @ state_vector + self.Bd @ control_input


    def step(self, state_vector:np.ndarray, control_input:np.ndarray, dt: float) -> np.ndarray:
        self.T += dt
        k1 = dt * self._dynamics(state_vector,control_input)
        k2 = dt * self._dynamics(state_vector+(0.5*k1),control_input)
        k3 = dt * self._dynamics(state_vector+(0.5*k2),control_input)
        k4 = dt * self._dynamics(state_vector+k3,control_input)

        new_state = state_vector + (k1 + 2*k2 + 2*k3 + k4) / 6 
        self.state_vector = new_state #+ self._wind_disturbance(self.T)
        return self.state_vector

