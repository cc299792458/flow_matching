import torch
import numpy as np
from tqdm import tqdm
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons

class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim)
        )
    
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))
    
    def step(self, x_t: Tensor, t_start: float, t_end: float) -> Tensor:
        t_start_tensor = torch.tensor(t_start, dtype=torch.float32).view(1, 1).expand(x_t.shape[0], 1)
        t_end_tensor = torch.tensor(t_end, dtype=torch.float32).view(1, 1).expand(x_t.shape[0], 1)
        return x_t + (t_end_tensor - t_start_tensor) * self(
            t=t_start_tensor + (t_end_tensor - t_start_tensor) / 2,
            x_t=x_t + self(x_t=x_t, t=t_start_tensor) * (t_end_tensor - t_start_tensor) / 2
        )

# Initialize and train model
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()


for _ in tqdm(range(10000)):
    x_1 = torch.concat(
        [torch.randn(100, 2) + torch.tensor([10, 10]), torch.randn(100, 2) + torch.tensor([10, -10])]
    )
    x_1_idx = torch.randperm(x_1.size(0))
    x_1 = x_1[x_1_idx, :]
    x_0 = torch.concat(
        [torch.randn(100, 2) + torch.tensor([-10, 10]), torch.randn(100, 2) + torch.tensor([-10, -10])]
    )
    x_0_idx = torch.randperm(x_0.size(0))
    x_0 = x_0[x_0_idx, :]
    t = torch.rand(len(x_1), 1)
    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    optimizer.zero_grad()
    loss_fn(flow(t=t, x_t=x_t), dx_t).backward()
    optimizer.step()

# Set up animation figure
fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
ax.set_title("Flow Matching Trajectory")
ax.grid(True)

# Initialize points
x_gif = torch.concat([torch.randn(100, 2) + torch.tensor([-10, 10]), torch.randn(100, 2) + torch.tensor([-10, -10])])
x_gif.requires_grad = False
scatter = ax.scatter(x_gif[:, 0].detach().numpy(), x_gif[:, 1].detach().numpy(), 
                    c=['blue'] * 100 + ['green'] * 100, s=50)
target_scatter = ax.scatter(x_1[:, 0].detach().numpy(), x_1[:, 1].detach().numpy(), 
                          c='red', s=50, marker='x', label='Targets')
ax.legend()

# Animation function
def init():
    global x_gif
    x_gif = torch.concat([torch.randn(100, 2) + torch.tensor([-10, 10]), torch.randn(100, 2) + torch.tensor([-10, -10])])
    x_gif.requires_grad = False
    scatter.set_offsets(x_gif.detach().numpy())
    return [scatter]

def update(frame):
    global x_gif
    t_start = (frame-1)/200 if frame > 0 else 0.0
    t_end = frame/200
    with torch.no_grad():
        x_gif = flow.step(x_t=x_gif, t_start=t_start, t_end=t_end)
    scatter.set_offsets(x_gif.detach().numpy())
    return [scatter]

# Create and display animation
ani = FuncAnimation(fig, update, frames=200, 
                   init_func=init, 
                   interval=50, blit=True)

plt.show()