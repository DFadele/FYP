import numpy as np
from numpy import sin, cos, tan
from scipy import integrate


class NLDroneModel:

    def __init__(self):
        self.m = 0.65
        self.Ix = 7.5e-3
        self.Iy = 7.5e-3
        self.Iz = 1.3e-2

        self.current_state = None

    def _dynamics(self, state_vector: np.ndarray, control_input: np.ndarray) -> np.ndarray:
       
        if control_input is None:
            control_input = np.zeros(4)
        if state_vector is None:
            print(state_vector)
            raise ValueError("State vector is None")
       
        x = state_vector[0]
        y = state_vector[1]
        z = state_vector[2]
        u = state_vector[3]
        v = state_vector[4]
        w = state_vector[5]
        phi = state_vector[6]
        theta = state_vector[7]
        psi = state_vector[8]
        p = state_vector[9]
        q = state_vector[10]
        r = state_vector[11]
        g = 9.81

        u1, u2, u3, u4 = control_input
        dx = u*(cos(theta)*cos(psi)) + v*(sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)) + w*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
        dy = u*(cos(theta)*sin(psi)) + v*(sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi)) + w*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
        dz = u*(sin(theta)) - v*(sin(phi)*cos(theta)) - w*(cos(phi)*cos(theta))
        du = r*v - q*w - g*sin(theta)
        dv = p*w - r*u + g*cos(theta)*sin(phi)
        dw = q*u - p*v + g*cos(theta)*cos(phi) - u1/self.m
        dphi = p + q*(sin(phi)*tan(theta)) + r*(cos(phi)*tan(theta))
        dtheta = q*cos(phi) - r*sin(phi)
        dpsi = (q*sin(phi) + r*cos(phi))/cos(theta)
        dp = (self.Iy-self.Iz)/self.Ix *q*r + u2/self.Ix
        dq = (self.Iz-self.Ix)/self.Iy *p*r + u3/self.Iy
        dr = (self.Ix-self.Iy)/self.Iz *p*q + u4/self.Iz


        return np.array([dx,dy,dz,du,dv,dw,dphi,dtheta,dpsi,dp,dq,dr])
    
    def integrate_dynamics2(self, state_vector:np.ndarray, control_input: np.ndarray, dt: float):
        if control_input is None:
            control_input = np.zeros(4)
        if state_vector is None:
            print(state_vector)
            raise ValueError("State vector is None")
        state_vector = state_vector.flatten('C')
        control_input = control_input.flatten('C')
        current_state = state_vector + dt*self._dynamics(state_vector,control_input)
        self.current_state = current_state
        return current_state


    def integrate_dynamics(self, state_vector:np.ndarray, control_input: np.ndarray, dt: float) -> np.ndarray:
        solver = integrate.RK45(lambda t, y: self._dynamics(y, control_input), 0, state_vector, dt)
        while solver.status == 'running':
            solver.step()
        self.current_state = solver.y
        return self.current_state

