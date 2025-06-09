import pybullet as p
import pybullet_data
import time, os
import numpy as np
import cvxpy as cp
from drone_model import DroneModel

np.set_printoptions(precision=3, suppress=True)

UDRF_PATH = os.path.join("gym-pybullet-drones", "gym_pybullet_drones", "assets", "cf2x.urdf")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)


drone = p.loadURDF(UDRF_PATH, [0, 0, 0])
print('Number of joints: ',p.getNumJoints(drone))


motor_positions = [
    [0.1, 0.1, 0],  # Motor 1 (Front Left)
    [-0.1, 0.1, 0],  # Motor 2 (Rear Left)
    [0.1, -0.1, 0],  # Motor 3 (Front Right)
    [-0.1, -0.1, 0]  # Motor 4 (Rear Right)
]

def update_camera(drone_id, camera_distance=1.5, yaw=50, pitch=-30):
    pos, _ = p.getBasePositionAndOrientation(drone_id)

    # Update camera position 
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=pos
    )

def compute_control(current_state, desired_state):
    # control logic
    n_states = 12
    n_inputs = 4
    N = 10
    C = np.array([
        [1,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0],
    ])
    predictor = DroneModel(Ts=1/240)
    #predictor.computeA(current_state)
    x = cp.Variable((n_states, N+1))
    u = cp.Variable((n_inputs, N))
    constraints = [x[:,0] == current_state]
    cost = 0
    Q = np.diag([1,1,1]*4/np.square([300,300,300,])) * 300
    R = np.diag([1,1,1,1]/np.square([40,3,3,1]))
    for k in range(N):
        cost += cp.quad_form((x[:,k] - desired_state), Q) + cp.quad_form(u[:,k],R)
        constraints += [
            x[:,k+1] == predictor.Ad @ x[:,k] + predictor.Bd @ u[:,k],
            #x[6:9,k] <= 10,
            #x[6:9,k] >= -10,
            #cp.norm(u[:,k], np.inf) <= 100
            ]
    objective = cp.Minimize(cost)    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    control_input = u[:,0].value
    return control_input

def apply_motor_forces(drone_id,control_input: np.ndarray):
    l = 0.1
    b = 1e-6
    d = 7.94e-12
    if control_input is not None:
        U1, U2, U3, U4 = control_input
    else:
        U1,U2,U3,U4 = [0.65*9.81,0,0,0]
    # Simple torque allocation
    F1 = (U1 + U2 / l + U3 / l - U4 / d) / 4
    F2 = (U1 - U2 / l + U3 / l + U4 / d) / 4
    F3 = (U1 + U2 / l - U3 / l + U4 / d) / 4
    F4 = (U1 - U2 / l - U3 / l - U4 / d) / 4
    motor_forces = [F1, F2, F3, F4]

    for i in range(4):
        p.applyExternalForce(
            objectUniqueId=drone_id,
            linkIndex=-1,
            forceObj=[0, 0, motor_forces[i]],  
            posObj=motor_positions[i],
            flags=p.LINK_FRAME 
        )

num = 0
desired_state = np.array([0,0,3,0,0,0,0,0,0,0,0,0])
while True:
    update_camera(drone)
    pos, orn = p.getBasePositionAndOrientation(drone)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    lin_vel, ang_vel = p.getBaseVelocity(drone)

    state = np.array([pos[0], pos[1], pos[2], roll, pitch, yaw, lin_vel[0], lin_vel[1], lin_vel[2], ang_vel[0], ang_vel[1], ang_vel[2]])
    #print(f"{state}, Iter: {num}\r", end="")
    control_input = compute_control(state, desired_state)  
    print(f"Control Input: {control_input} Iteration: {num}\r",end="")
    apply_motor_forces(drone, control_input)
    p.stepSimulation()
    time.sleep(1/240)
    num += 1
