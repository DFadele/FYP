import cvxpy as cp
import numpy as np
from drone_model import DroneModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import trajectory
import matplotlib.animation as ani
from pyplot3d.uav import Uav
from pyplot3d.utils import ypr_to_R

plt.rcParams['axes.grid'] = True
N = 18
n_states = 12
n_inputs = 4

x0 = np.array([0,0,15,0,0,0])
x_ref = trajectory.sine_wave(220, 15, 0, 20, 0)

C = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0],
])

Q = np.diag([1,1,1]/np.square([300,300,300])) * 300
R = np.diag([1,1,1,1]/np.square([40,3,3,1]))

 
STEPS = 220
plant = DroneModel(Ts=0.1)
predictor = DroneModel(Ts=0.1)
plant_states = np.zeros((n_states,STEPS+1))
plant_states[:,0] = np.concatenate((x0,[0,0,0,0,0,0]))
inputs = np.zeros((n_inputs,STEPS))
for t in tqdm(range(STEPS),desc='Running Simulation...'):
    current_state = plant_states[:,t]
    if t == 0:
        predictor.computeA(current_state)
        plant.computeA(current_state)
    x = cp.Variable((n_states, N+1))
    u = cp.Variable((n_inputs, N))
    constraints = [x[:,0] == current_state]
    cost = 0
    for k in range(N):
        ref_k = x_ref[:, min(t + k, x_ref.shape[1] - 1)]
        cost += cp.quad_form(C @ x[:,k] - ref_k, Q) + cp.quad_form(u[:,k],R)
        constraints += [
            x[:,k+1] == predictor.discrete_dynamics(x[:,k],u[:,k]),
            x[6:9,k] <= 10,
            x[6:9,k] >= -10,
            x[6:9,k] <= np.pi/2,
            x[6:9,k] >= -np.pi/2,
            cp.norm(u[:,k], np.inf) <= 5
            ]
    objective = cp.Minimize(cost)    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status not in ["infeasible", "unbounded"]:
        optimal_u = u[:,0].value
        plant_states[:,t+1] = plant.step(current_state, optimal_u,0.1)
        inputs[:,t] = optimal_u
    else:
        print(f"\nProblem at step {t} is {prob.status}")
        optimal_u = inputs[:,t-1]
        plant_states[:,t+1] = plant.step(current_state, optimal_u,0.1)
        inputs[:,t] = optimal_u
        continue

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(plant_states[0,:], plant_states[1,:], plant_states[2,:], label='Drone Position')
ax.plot(x_ref[0,:], x_ref[1,:], x_ref[2,:], color='r', label='Reference Position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
#plt.show()

fig = plt.figure(figsize=(16, 6))

# Left plot: 3D Drone Trajectory Animation
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim([np.min(x_ref[0, :]) - 1, np.max(x_ref[0, :]) + 1])
ax1.set_ylim([np.min(x_ref[1, :]) - 1, np.max(x_ref[1, :]) + 1])
ax1.set_zlim([np.min(x_ref[2, :]) - 1, np.max(x_ref[2, :]) + 1])
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_zlabel('Z Position (m)')
#ax1.set_title('3D Drone Trajectory Animation')

# Plot trajectory path
ax1.plot(x_ref[0, :], x_ref[1, :], x_ref[2, :], '--', color='gray', label="Reference Path")

drone = ax1.plot([], [], [], 'rx', markersize=10, label="Drone Position")[0]
path = ax1.plot([], [], [], 'b-')[0]
x_trail, y_trail, z_trail = [], [], []

uav = Uav(ax1, arm_length=0.5)


def update_drone(num):
    if num == 0:
        x_trail.clear()
        y_trail.clear()
        z_trail.clear()
    x_trail.append(plant_states[0, num])
    y_trail.append(plant_states[1, num])
    z_trail.append(plant_states[2, num])

    drone.set_data([plant_states[0, num]], [plant_states[1, num]])
    drone.set_3d_properties([plant_states[2, num]])

    path.set_data(x_trail, y_trail)
    path.set_3d_properties(z_trail)

    #ax1.set_xlim([x_ref[0, num] - 2, x_ref[0, num] + 2])
    ax1.set_ylim([x_ref[1, num] - 6, x_ref[1, num] + 6])
    ax1.set_zlim([x_ref[2, num] - 2, x_ref[2, num] + 2])
    return drone, path


# Right plot: Control Inputs as 4 Subplots
control_axes = [fig.add_subplot(4, 2, i * 2 + 2) for i in range(4)]
control_lines = []

for i, ax in enumerate(control_axes):
    line, = ax.plot([], [], label=f'$u_{i+1}$')
    control_lines.append(line)
    ax.set_xlim(0, STEPS * plant.Ts)
    if i == 0:
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylim(np.min(inputs[i]), np.max(inputs[i]))
    ax.set_ylabel(f'$u_{i+1}$')
    ax.legend()
    if i == 3:
        ax.set_xlabel('Time (s)')
fig.tight_layout()
time = np.arange(0, STEPS * plant.Ts, plant.Ts)  # Time in seconds


def update_control(num):
    for i in range(4):
        control_lines[i].set_data(time[:num], inputs[i, :num])
    return control_lines

# Combine animations
def update_combined(num):
    update_drone(num)
    update_control(num)
    return drone, path, *control_lines


animation = ani.FuncAnimation(fig, update_combined, frames=range(STEPS), blit=False, interval=0.1 * 1000)

writer = ani.PillowWriter(fps=10)
animation.save('combined_animation.gif', writer=writer)
plt.show()



