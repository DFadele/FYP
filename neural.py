import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nonlinear_drone import NLDroneModel 

# Hyperparameters
dt = 0.005
epochs = 100
batch_size = 256
learning_rate = 1e-3


plant = NLDroneModel()
x_data = []
u_data = []
x_next_data = []

STEPS = 5000
# synthetic dataset
for _ in range(STEPS):
    x = np.random.uniform(low=-1.0, high=1.0, size=(12,))
    u = np.random.uniform(low=[0, -1, -1, -1], high=[15, 1, 1, 1], size=(4,))
    x_next = plant.integrate_dynamics(x, u, dt)

    x_data.append(x)
    u_data.append(u)
    x_next_data.append(x_next)

x_data = np.array(x_data)
u_data = np.array(u_data)
x_next_data = np.array(x_next_data)


X = torch.tensor(np.hstack([x_data, u_data]), dtype=torch.float32)
Y = torch.tensor(x_next_data, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class DroneDynamicsNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )

    def forward(self, xu):
        return self.net(xu)

if __name__ == '__main__':
    model = DroneDynamicsNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()


    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}\r",end="")

   
    torch.save(model.state_dict(), "drone_dynamics_model.pth")
