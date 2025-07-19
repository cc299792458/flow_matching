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
    
    def step(self, x_t: Tensor, t_start: float, t_end: float, condition: Tensor, guidance: float = 1.0, 
            unconditional: bool = False) -> Tensor:
        """
        Generate next step with optional unconditional generation
        
        Args:
            x_t: Current state tensor
            t_start: Starting time
            t_end: Ending time
            condition: Conditional input tensor
            guidance: Guidance scale factor
            unconditional: Flag for unconditional generation
        """
        batch_size = x_t.size(0)
        t_mid = t_start + (t_end - t_start) / 2
        
        # Create time tensors
        t_start_tensor = torch.full((batch_size, 1), t_start, dtype=torch.float32)
        t_mid_tensor = torch.full((batch_size, 1), t_mid, dtype=torch.float32)
        
        # Use null embedding if unconditional generation requested
        if unconditional:
            # Override condition with null embedding during sampling
            condition = self.null_embed.expand_as(condition)
            
        # First step: midpoint prediction (with guidance)
        k1 = self(
            t=t_start_tensor, 
            x_t=x_t, 
            condition=condition,
            guidance=guidance,
            use_guidance=(not unconditional)  # Disable guidance for unconditional
        )
        x_mid = x_t + k1 * (t_end - t_start) / 2
        
        # Second step: final prediction using midpoint (with guidance)
        k2 = self(
            t=t_mid_tensor,
            x_t=x_mid,
            condition=condition,
            guidance=guidance,
            use_guidance=(not unconditional)  # Disable guidance for unconditional
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

plt.legend()
plt.tight_layout()
plt.show()

# First, we'll run the entire simulation without animation to compute the final variances
num_visualize = 500
num_frames = 200
half = num_visualize // 2

# Initialize particle states
x_conditional = std * torch.randn(num_visualize, 2)
x_unconditional = std * torch.randn(num_visualize, 2)
conditions = torch.cat([
    target_1.repeat(half, 1),
    target_2.repeat(num_visualize - half, 1)
])

# Define storage for final particle states
final_conditional = None
final_unconditional = None

# Run simulation to get final variances
print("Computing final variances...")
for frame in tqdm(range(num_frames)):
    t_start = frame / num_frames
    t_end = (frame + 1) / num_frames
    
    with torch.no_grad():
        # Update conditional particles
        x_conditional = flow.step(
            x_t=x_conditional,
            t_start=t_start,
            t_end=t_end,
            condition=conditions,
            guidance=1.5
        )
        
        # Update unconditional particles
        x_unconditional = flow.step(
            x_t=x_unconditional,
            t_start=t_start,
            t_end=t_end,
            condition=conditions,
            unconditional=True
        )
    
    # Store final states on last frame
    if frame == num_frames - 1:
        final_conditional = x_conditional.clone()
        final_unconditional = x_unconditional.clone()

# Calculate final variances
print("\nFinal Standard Deviations:")
print("-" * 30)

# Conditional group metrics
cond_groups = {
    '[5,5]': final_conditional[:half],
    '[5,-5]': final_conditional[half:]
}

for group, tensor in cond_groups.items():
    std_x = torch.std(tensor[:, 0]).item()
    std_y = torch.std(tensor[:, 1]).item()
    print(f"{group} Group (Conditional):")
    print(f"  X-axis: {std_x:.4f}")
    print(f"  Y-axis: {std_y:.4f}")

# Unconditional metrics
uncond_std_x = torch.std(final_unconditional[:, 0]).item()
uncond_std_y = torch.std(final_unconditional[:, 1]).item()
print(f"\nUnconditional Group:")
print(f"  X-axis: {uncond_std_x:.4f}")
print(f"  Y-axis: {uncond_std_y:.4f}")

print("-" * 30)
print("Starting visualization animation...\n")

# Now create the visualization animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Flow Matching - Conditional vs Unconditional")
ax.grid(True)

# Reinitialize particle states for visualization
x_cond_viz = std * torch.randn(num_visualize, 2)
x_uncond_viz = std * torch.randn(num_visualize, 2)

# Color coding: 
# Blue = [5,5] conditional | Green = [5,-5] conditional | Gray = Unconditional
colors_cond = ['blue'] * half + ['green'] * (num_visualize - half)
color_uncond = 'gray'

# FIX: Initialize scatter plots with starting positions
# Provide initial positions to avoid the "size 0" error
scatter_cond = ax.scatter(
    x_cond_viz[:, 0].detach().numpy(),
    x_cond_viz[:, 1].detach().numpy(),
    c=colors_cond, 
    s=30, 
    alpha=0.6, 
    label='Conditional'
)

scatter_uncond = ax.scatter(
    x_uncond_viz[:, 0].detach().numpy(),
    x_uncond_viz[:, 1].detach().numpy(),
    c=color_uncond, 
    s=30, 
    alpha=0.6, 
    label='Unconditional'
)

# Plot target points
target_scatter = ax.scatter(
    [target_1[0], target_2[0]], 
    [target_1[1], target_2[1]],
    c=['red', 'purple'], 
    s=200, 
    marker='*', 
    edgecolor='black'
)

# Set axis limits
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.legend()

def init():
    """Initialize animation with random samples"""
    global x_cond_viz, x_uncond_viz
    x_cond_viz = std * torch.randn(num_visualize, 2)
    x_uncond_viz = std * torch.randn(num_visualize, 2)
    
    # Set initial positions
    scatter_cond.set_offsets(x_cond_viz.numpy())
    scatter_uncond.set_offsets(x_uncond_viz.numpy())
    return [scatter_cond, scatter_uncond]

def update(frame):
    """Update function for each animation frame"""
    global x_cond_viz, x_uncond_viz
    
    t_start = frame / num_frames
    t_end = (frame + 1) / num_frames
    
    with torch.no_grad():
        # Update conditional particles
        x_cond_viz = flow.step(
            x_t=x_cond_viz,
            t_start=t_start,
            t_end=t_end,
            condition=conditions,
            guidance=1.5
        )
        
        # Update unconditional particles
        x_uncond_viz = flow.step(
            x_t=x_uncond_viz,
            t_start=t_start,
            t_end=t_end,
            condition=conditions,
            unconditional=True
        )
    
    # Update visualization
    scatter_cond.set_offsets(x_cond_viz.numpy())
    scatter_uncond.set_offsets(x_uncond_viz.numpy())
    ax.set_title(f"Generation Process - Step {frame+1}/{num_frames}")
    
    return [scatter_cond, scatter_uncond]

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, 
                   init_func=init, 
                   interval=50, blit=False)

plt.show()