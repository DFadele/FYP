import torch
import numpy as np
from neural import DroneDynamicsNN

model = DroneDynamicsNN()
model.load_state_dict(torch.load("drone_dynamics_model.pth"))
model.eval()


def linearise_nn(x_T, u_T):
    
    x = torch.tensor(x_T, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    u = torch.tensor(u_T, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    xu = torch.cat([x,u],dim=1)

    y = model(xu)

    A = torch.zeros((12,12))
    B = torch.zeros((12,4))

    for i in range(12):
        grad_x = torch.autograd.grad(y[0, i], x, retain_graph=True, allow_unused=True)[0]
        grad_u = torch.autograd.grad(y[0, i], u, retain_graph=True, allow_unused=True)[0]

        if grad_x is not None:
            A[i, :] = grad_x[0]
        if grad_u is not None:
            B[i, :] = grad_u[0]

        #x.grad.zero_()
        #u.grad.zero_()

    return A.detach().numpy(), B.detach().numpy()