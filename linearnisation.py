import sympy as sp
from sympy import sin, cos, tan

x, y, z, u, v, w = sp.symbols('x y z u v w')
phi, theta, psi = sp.symbols('phi theta psi')
p, q, r = sp.symbols('p q r')
u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')

m, g = sp.symbols('m g')
Ix, Iy, Iz = sp.symbols('I_x I_y I_z')

state = sp.Matrix([x,y,z,u,v,w,phi,theta,psi,p,q,r])
inputs = sp.Matrix([u1,u2,u3,u4])

# Translational Dynamics
dx = u*(cos(theta)*cos(psi)) + v*(sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi)) + w*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))
dy = u*(cos(theta)*sin(psi)) + v*(sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi)) + w*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))
dz = u*(sin(theta)) - v*(sin(phi)*cos(theta)) - w*(cos(phi)*cos(theta))
du = r*v - q*w - g*sin(theta)
dv = p*w - r*u + g*cos(theta)*sin(phi)
dw = q*u - p*v + g*cos(theta)*cos(phi) - u1/m

# Rotational Dynamics
dphi = p + q*(sin(phi)*tan(theta)) + r*(cos(phi)*tan(theta))
dtheta = q*cos(phi) - r*sin(phi)
dpsi = (q*sin(phi) + r*cos(phi))/cos(theta)
dp = (Iy-Iz)/Ix *q*r + u2/Ix
dq = (Iz-Ix)/Iy *p*r + u3/Iy
dr = (Ix-Iy)/Iz *p*q + u4/Iz

f = sp.Matrix([dx,dy,dz,du,dv,dw,dphi,dtheta,dpsi,dp,dq,dr])

Asym = f.jacobian(state)
Bsym = f.jacobian(inputs)

