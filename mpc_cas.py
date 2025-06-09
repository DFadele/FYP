import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from casadiNN import nn_func  # Your CasADi NN function
from tqdm import tqdm

N = 2
dt = 0.005
n_state = 12
n_input = 4
T_total = 1
steps = int(T_total / dt)

Q = np.diag([10.0, 10.0, 10.0] + [0.1]*(n_state - 3))
R = np.diag([0.1]*n_input)
x_ref = np.array([1.0, 0.0, -1.0] + [0.0]*(n_state - 3))

x0 = ca.MX.sym("x0", n_state)
U = ca.MX.sym("U", n_input, N)
X = [x0]
cost = 0
g = []

for k in range(N):
    u_k = U[:, k]
    x_k = X[-1]
    xu_k = ca.vertcat(x_k, u_k)
    x_next = nn_func(xu_k)
    X.append(x_next)

    err = x_next - x_ref
    cost += ca.mtimes([err.T, Q, err]) + ca.mtimes([u_k.T, R, u_k])

    g.append(u_k[0])       # thrust ≥ 0
    g.append(10 - u_k[0])  # thrust ≤ 10

opt_vars = ca.reshape(U, -1, 1)
nlp = {'x': opt_vars, 'f': cost, 'g': ca.vertcat(*g), 'p': x0}
solver = ca.nlpsol("solver", "ipopt", nlp,{
    "ipopt.print_level":0,
    "print_time":False,
    "verbose":False,
    "ipopt.sb":"yes",
    "ipopt.max_iter":100,
    "ipopt.tol":1e-2,
    "ipopt.acceptable_tol":1e-1,
    "ipopt.linear_solver":"mumps"
})

# Closed-Loop Simulation 
x_log = np.zeros((n_state, steps+1))
u_log = np.zeros((n_input, steps))
x_log[:, 0] = np.zeros(n_state)

for t in tqdm(range(steps),desc="Running Simulation..."):
    x_t = x_log[:, t]
    sol = solver(
        x0=np.zeros((n_input * N, 1)),
        p=x_t,
        lbg=0.0,
        ubg=10.0
    )
    u_opt = sol['x'].full().reshape(n_input, N)
    u_apply = u_opt[:, 0]
    u_log[:, t] = u_apply

    x_next = nn_func(np.concatenate([x_t, u_apply])).full().flatten()
    x_log[:, t+1] = x_next


plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
t = np.arange(steps+1)*dt
plt.plot(t, x_log[0], label='x')
plt.plot(t, x_log[1], label='y')
plt.plot(t, x_log[2], label='z')
plt.axhline(x_ref[0], linestyle='--', color='r', label='x_ref')
plt.axhline(x_ref[2], linestyle='--', color='g', label='z_ref')
plt.ylabel("Position")
plt.title("Closed-loop MPC using NN Dynamics")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(steps)*dt, u_log[0], label='Thrust')
plt.ylabel("Control")
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.show()
