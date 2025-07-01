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

# Store loss values during training
loss_history = []

num_train = 1000
for _ in tqdm(range(1_000)):
    x_1 = torch.concat(
        [torch.randn(num_train, 2) + torch.tensor([10, 10]), torch.randn(num_train, 2) + torch.tensor([10, -10])]
    )
    x_1_idx = torch.randperm(x_1.size(0))
    x_1 = x_1[x_1_idx, :]
    x_0 = torch.concat(
        [torch.randn(num_train, 2) + torch.tensor([-10, 10]), torch.randn(num_train, 2) + torch.tensor([-10, -10])]
    )
    x_0_idx = torch.randperm(x_0.size(0))
    x_0 = x_0[x_0_idx, :]
    t = torch.rand(len(x_1), 1)
    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    optimizer.zero_grad()
    loss = loss_fn(flow(t=t, x_t=x_t), dx_t)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())  # Record current loss

# ==================== Plot Loss Curve ====================
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label='Training Loss', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Add smoothed version (optional)
window_size = 100
smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
plt.plot(smoothed_loss, label=f'Smoothed (window={window_size})', color='red', linewidth=2)

plt.legend()
plt.tight_layout()
plt.show()

# Set up animation figure
fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
ax.set_title("Flow Matching Trajectory")
ax.grid(True)

# Initialize points
num_visualize = 500
x_gif = torch.concat([torch.randn(num_visualize, 2) + torch.tensor([-10, 10]), 
                      torch.randn(num_visualize, 2) + torch.tensor([-10, -10])
                    ])
x_gif.requires_grad = False
scatter = ax.scatter(x_gif[:, 0].detach().numpy(), x_gif[:, 1].detach().numpy(), 
                    c=['blue'] * num_visualize + ['green'] * num_visualize, s=50)
target_scatter = ax.scatter(x_1[:num_visualize, 0].detach().numpy(), x_1[:num_visualize, 1].detach().numpy(), 
                          c='red', s=50, marker='x', label='Targets')
ax.legend()

# Animation function
def init():
    global x_gif
    x_gif = torch.concat([torch.randn(num_visualize, 2) + torch.tensor([-10, 10]), 
                          torch.randn(num_visualize, 2) + torch.tensor([-10, -10])
                        ])
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