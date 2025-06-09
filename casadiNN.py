import casadi as ca
import torch
import numpy as np
from neural import DroneDynamicsNN  

# === Load Trained PyTorch Model ===
model = DroneDynamicsNN()
model.load_state_dict(torch.load("drone_dynamics_model.pth"))
model.eval()

# === Extract Weights and Biases ===
weights = []
for param in model.parameters():
    weights.append(param.detach().numpy())

# Assume: Linear(16→64) → ReLU → Linear(64→64) → ReLU → Linear(64→12)
W1, b1 = weights[0], weights[1].reshape(-1, 1)
W2, b2 = weights[2], weights[3].reshape(-1, 1)
W3, b3 = weights[4], weights[5].reshape(-1, 1)

# === Define CasADi Equivalent ===
def relu(x):
    return ca.fmax(0, x)

def casadi_nn(xu):
    z1 = relu(W1 @ xu + b1)
    z2 = relu(W2 @ z1 + b2)
    out = W3 @ z2 + b3
    return out

# === Create a CasADi Function ===
xu = ca.MX.sym("xu", 16)  # [12 state + 4 input]
nn_output = casadi_nn(xu)
nn_func = ca.Function("nn_dynamics", [xu], [nn_output])

# === (Optional) Test Output ===
if __name__ == "__main__":
    dummy_xu = np.random.randn(16)
    print("NN prediction:", nn_func(dummy_xu))
