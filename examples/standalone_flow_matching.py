import torch
import numpy as np
from tqdm import tqdm
from torch import nn, Tensor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.dim = dim
        # Learnable null embedding for unconditional generation
        self.null_embed = nn.Parameter(torch.zeros(1, dim))
        
        # Network architecture remains the same
        self.net = nn.Sequential(
            nn.Linear(dim + 1 + dim, h), nn.ELU(),  # +dim for condition
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim)
        )
    
    def forward(self, t: Tensor, x_t: Tensor, condition: Tensor, guidance: float = 1.0, use_guidance: bool = False) -> Tensor:
        # During training: 10% chance to use null embedding (unconditional)
        if self.training:
            mask = torch.rand(x_t.size(0), 1) < 0.1
            condition = torch.where(mask, self.null_embed.expand_as(condition), condition)
        
        # Regular forward pass without guidance
        if not use_guidance:
            return self.net(torch.cat((t, x_t, condition), -1))
        
        # Guidance forward pass:
        # 1. Conditional prediction
        cond_output = self.net(torch.cat((t, x_t, condition), -1))
        # 2. Unconditional prediction (using null embedding)
        uncond_output = self.net(torch.cat((t, x_t, self.null_embed.expand_as(condition)), -1))
        # 3. Apply guidance mixing
        return uncond_output + guidance * (cond_output - uncond_output)
    
    def step(self, x_t: Tensor, t_start: float, t_end: float, condition: Tensor, guidance: float = 1.0) -> Tensor:
        batch_size = x_t.size(0)
        t_mid = t_start + (t_end - t_start) / 2
        
        # Create time tensors
        t_start_tensor = torch.full((batch_size, 1), t_start, dtype=torch.float32)
        t_mid_tensor = torch.full((batch_size, 1), t_mid, dtype=torch.float32)
        
        # First step: midpoint prediction (with guidance)
        k1 = self(
            t=t_start_tensor, 
            x_t=x_t, 
            condition=condition,
            guidance=guidance,
            use_guidance=True
        )
        x_mid = x_t + k1 * (t_end - t_start) / 2
        
        # Second step: final prediction using midpoint (with guidance)
        k2 = self(
            t=t_mid_tensor,
            x_t=x_mid,
            condition=condition,
            guidance=guidance,
            use_guidance=True
        )
        return x_t + k2 * (t_end - t_start)

# Initialize and train model (same structure)
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()

# Store loss values during training
loss_history = []

# Training parameters
num_train = 1
std = 1
batchsize = 500

# Define two target conditions
num_target = 2
target_1 = torch.tensor([5., 5.])
target_2 = torch.tensor([5., -5.])

for _ in tqdm(range(1_000)):
    # Randomly assign conditions (50% target_1, 50% target_2)
    condition_mask = torch.rand(batchsize, 1) < 0.5
    x_1 = torch.where(condition_mask, 
                     target_1.repeat(batchsize, 1), 
                     target_2.repeat(batchsize, 1))
    
    # Shuffle samples
    idx = torch.randperm(batchsize)
    x_1 = x_1[idx]
    
    # Initial noise samples
    x_0 = std * torch.randn(batchsize, 2)
    
    # Create timesteps and interpolated points
    t = torch.rand(batchsize, 1)
    x_t = (1 - t) * x_0 + t * x_1
    
    # Compute true gradient direction
    dx_t = x_1 - x_0
    
    # Training step
    optimizer.zero_grad()
    output = flow(t=t, x_t=x_t, condition=x_1)
    loss = loss_fn(output, dx_t)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# Plot loss curve (unchanged)
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label='Training Loss', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

window_size = 100
smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
plt.plot(smoothed_loss, label=f'Smoothed (window={window_size})', color='red', linewidth=2)

plt.legend()
plt.tight_layout()
plt.show()

# Animation setup with two subplots: left for trajectories, right for std metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.set_title("Flow Matching with Conditional Guidance (λ=1.5)")
ax1.grid(True)

# Right plot for standard deviation tracking (X and Y separately)
ax2.set_title("Standard Deviation by Condition Group (X/Y)")
ax2.set_xlabel("Generation Step")
ax2.set_ylabel("Standard Deviation")
ax2.grid(True)

# Initialize samples for visualization
num_visualize = 500  # Number of particles to visualize
x_gif = std * torch.randn(num_visualize, 2)  # Initial random samples
x_gif.requires_grad = False

# Storage for standard deviation history
std_history = []

# Create condition assignments (half [5,5], half [5,-5])
half = num_visualize // 2
conditions = torch.cat([
    target_1.repeat(half, 1),
    target_2.repeat(num_visualize - half, 1)
])
condition_groups = ['[5,5]'] * half + ['[5,-5]'] * (num_visualize - half)

# Color coding: blue for [5,5], green for [5,-5]
colors = ['blue'] * half + ['green'] * (num_visualize - half)
scatter = ax1.scatter(x_gif[:, 0].detach().numpy(), x_gif[:, 1].detach().numpy(), 
                     c=colors, s=50, alpha=0.6)

# Plot target points with star markers
target_scatter = ax1.scatter(
    [target_1[0], target_2[0]], [target_1[1], target_2[1]],
    c=['red', 'purple'], s=200, marker='*', edgecolor='black', 
    label=['Target (5,5)', 'Target (5,-5)']
)

# Set axis limits for main plot
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.legend(loc='upper right')

# Initialize std plot lines for each condition and axis
steps = []
std_lines = {
    '[5,5]_x': ax2.plot([], [], 'b-', label='[5,5] X')[0],  # Solid blue for [5,5] X
    '[5,5]_y': ax2.plot([], [], 'b--', label='[5,5] Y')[0],  # Dashed blue for [5,5] Y
    '[5,-5]_x': ax2.plot([], [], 'g-', label='[5,-5] X')[0],  # Solid green for [5,-5] X
    '[5,-5]_y': ax2.plot([], [], 'g--', label='[5,-5] Y')[0]  # Dashed green for [5,-5] Y
}
ax2.legend()
ax2.set_xlim(0, 200)  # 200 frames total
ax2.set_ylim(0, 5)  # Expected std range (adjust as needed)

# Track animation completion status
animation_complete = False
final_std_values = None

def init():
    """Initialize animation with random samples"""
    global x_gif, std_history, animation_complete
    x_gif = std * torch.randn(num_visualize, 2)
    x_gif.requires_grad = False
    scatter.set_offsets(x_gif.detach().numpy())
    std_history = []
    animation_complete = False
    return [scatter]

def update(frame):
    """Update function for each animation frame"""
    global x_gif, std_history, animation_complete, final_std_values
    
    # Calculate current time step
    t_start = (frame - 1) / 200 if frame > 0 else 0.0
    t_end = frame / 200
    
    # Generate next step with guidance
    with torch.no_grad():
        x_gif = flow.step(
            x_t=x_gif,
            t_start=t_start,
            t_end=t_end,
            condition=conditions,
            guidance=1.5
        )
    
    # Calculate standard deviations per condition group
    group_data = {
        '[5,5]': {'x': [], 'y': []},
        '[5,-5]': {'x': [], 'y': []}
    }
    
    # Group samples by their assigned condition
    for i in range(num_visualize):
        group = condition_groups[i]
        group_data[group]['x'].append(x_gif[i, 0].item())
        group_data[group]['y'].append(x_gif[i, 1].item())
    
    # Store metrics for current frame
    current_metrics = {'step': frame}
    for group in group_data:
        current_metrics[f'{group}_x'] = torch.std(torch.tensor(group_data[group]['x'])).item()
        current_metrics[f'{group}_y'] = torch.std(torch.tensor(group_data[group]['y'])).item()
    
    std_history.append(current_metrics)
    
    # Mark completion on final frame
    if frame == 199:
        animation_complete = True
        final_std_values = current_metrics
    
    # Update visualization
    scatter.set_offsets(x_gif.detach().numpy())
    ax1.set_title(f"Conditional Generation (λ=1.5) - Step {frame+1}/200")
    
    # Update std plot data
    steps = [s['step'] for s in std_history]
    for key in std_lines:
        std_lines[key].set_data(steps, [s[key] for s in std_history])
    
    # Auto-scale y-axis based on observed std values
    all_stds = [v for s in std_history for k,v in s.items() if k != 'step']
    current_max_std = max(all_stds) if all_stds else 1.0
    ax2.set_ylim(0, current_max_std * 1.1)
    
    return [scatter]

# Create animation with 200 frames
ani = FuncAnimation(fig, update, frames=200, 
                   init_func=init, 
                   interval=50, blit=False)

plt.show()

# Print final metrics after animation completes
if std_history:
    if animation_complete:
        print("\nAnimation completed successfully. Final Standard Deviations:")
    else:
        print("\nWarning: Animation interrupted. Partial results:")
    
    for group in ['[5,5]', '[5,-5]']:
        print(f"\n{group} Group:")
        print(f"  X-axis std: {final_std_values[f'{group}_x']:.4f}")
        print(f"  Y-axis std: {final_std_values[f'{group}_y']:.4f}")
    
    # Check for convergence
    if animation_complete:
        last_quarter = len(std_history) // 4
        last_stds = [s['[5,5]_x'] for s in std_history[-last_quarter:]]
        if max(last_stds) - min(last_stds) > 0.1:
            print("\nNote: X-axis std for [5,5] shows possible non-convergence")