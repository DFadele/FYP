import numpy as np
import matplotlib.pyplot as plt


def sine_wave(N:int, alt: float, start_x: float, end_x: float, y_pos: float):
    x_ref = np.zeros((3,N))
    X = np.linspace(start_x,end_x,N)
    Y = 5*np.sin(X) + y_pos
    Z = np.ones(N) * alt
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def straight_line(N:int, alt: float, start_x: float, end_x: float, y_pos: float):
    x_ref = np.zeros((3,N))
    X = np.linspace(start_x,end_x,N)
    Y = np.ones(N) * y_pos
    Z = np.ones(N) * alt
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref


def circular(N:int, alt: float, radius: float):
    x_ref = np.zeros((3,N))
    Z = np.ones(N) * alt
    theta = np.linspace(0,2*np.pi,N)
    X = np.cos(theta) * radius
    Y = np.sin(theta) * radius
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def helix(N:int, radius: float, start_z: float, end_z: float, N_rotations: int):
    x_ref = np.zeros((3,N))
    Z = np.linspace(start_z,end_z,N)
    theta = np.linspace(0,N_rotations*2*np.pi,N)
    X = np.cos(theta) * radius
    Y = np.sin(theta) * radius
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def figure8(N:int, alt: float, radius: float):
    x_ref = np.zeros((3,N))
    Z = np.ones(N) * alt
    theta = np.linspace(0,4*np.pi,N)
    X = np.sin(theta) * radius
    Y = np.sin(2*theta) * radius
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def spiral(N:int, start_z: float, end_z: float):
    x_ref = np.zeros((3,N))
    Z = np.linspace(start_z,end_z,N)
    X = Z * np.sin(3*Z) 
    Y = Z* np.cos(3*Z)
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def square(N:int, alt: float, side_length: float):
    x_ref = np.zeros((3,N))
    Z = np.ones(N) * alt
    X = np.concatenate((np.linspace(0, side_length, N//4), 
                        np.ones(N//4) * side_length,
                        np.linspace(side_length, 0, N//4),
                        np.zeros(N//4)))
    Y = np.concatenate((np.zeros(N//4), 
                        np.linspace(0, side_length, N//4),
                        np.ones(N//4) * side_length,
                        np.linspace(side_length, 0, N//4)))
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def zigzag(N:int, alt: float, amplitude: float, frequency: float):
    x_ref = np.zeros((3,N))
    Z = np.ones(N) * alt
    X = np.linspace(0, 10, N)
    Y = amplitude * np.sign(np.sin(frequency * X))
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def step_change(N:int, alt: float, step_size: float):
    x_ref = np.zeros((3,N))
    Z = np.ones(N) * alt
    X = np.linspace(0, 10, N)
    Y = np.zeros(N)
    for i in range(1, N):
        if i % step_size == 0:
            Y[i:] += 1
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def random_path(N:int, alt: float):
    x_ref = np.zeros((3,N))
    Z = np.ones(N) * alt
    X = np.linspace(0, 10, N)
    Y = np.random.rand(N) * 10
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

def setpoint(N:int, alt: float, x_pos: float, y_pos: float):
    x_ref = np.zeros((3,N))
    Z = np.ones(N) * alt
    X = np.ones(N) * x_pos
    Y = np.ones(N) * y_pos
    x_ref[0,:] = X
    x_ref[1,:] = Y
    x_ref[2,:] = Z
    return x_ref

if __name__ == '__main__':
    x_ref = sine_wave(200, 15, 0, 20, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_ref[0,:], x_ref[1,:], x_ref[2,:], color='r', label='Reference Position')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
