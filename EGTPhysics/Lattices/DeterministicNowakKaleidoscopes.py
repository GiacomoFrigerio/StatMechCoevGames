import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d

# Constants
C, D = 1, 0
R, S, P = 1, 0, 0
L = 199
steps = 500
T = 1.95
color_map = colors.ListedColormap(['blue', 'red', 'yellow', 'green'])

# Initialize grid with one defector in center
def initialize_grid():
    grid = np.ones((L, L), dtype=int)
    grid[L // 2, L // 2] = D
    return grid

# Compute payoff grid
def compute_payoff(grid, T):
    coop = (grid == C).astype(int)
    defe = (grid == D).astype(int)
    kernel = np.ones((3, 3))
    coop_sum = convolve2d(coop, kernel, mode='same', boundary='fill', fillvalue=0)
    defe_sum = convolve2d(defe, kernel, mode='same', boundary='fill', fillvalue=0)
    return coop * (R * coop_sum + S * defe_sum) + defe * (T * coop_sum + P * defe_sum)

# Update grid based on best neighbor (including self)
def update(grid, T):
    payoff = compute_payoff(grid, T)
    best_payoff = payoff.copy()
    best_strategy = grid.copy()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted_grid = np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
            shifted_payoff = np.roll(np.roll(payoff, dx, axis=0), dy, axis=1)

            # Apply fixed boundary condition
            if dx == -1:
                shifted_payoff[-1, :] = -np.inf
            elif dx == 1:
                shifted_payoff[0, :] = -np.inf
            if dy == -1:
                shifted_payoff[:, -1] = -np.inf
            elif dy == 1:
                shifted_payoff[:, 0] = -np.inf

            mask = shifted_payoff > best_payoff
            best_payoff[mask] = shifted_payoff[mask]
            best_strategy[mask] = shifted_grid[mask]
    return best_strategy

# Color transitions
def transition_colors(prev, curr):
    colors = np.zeros_like(curr)
    colors[(prev == C) & (curr == C)] = 0  # Blue
    colors[(prev == D) & (curr == D)] = 1  # Red
    colors[(prev == C) & (curr == D)] = 2  # Yellow
    colors[(prev == D) & (curr == C)] = 3  # Green
    return colors

# Run simulation
grid = initialize_grid()
frames = []
for _ in range(steps):
    prev = grid.copy()
    grid = update(grid, T)
    frames.append(transition_colors(prev, grid))

# Plot selected symmetric steps
selected = [0, 20, 40, 60, 90, 120, 150, 179]
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"Evolutionary Kaleidoscope in a {L} x {L} lattice", fontsize=16, y=1.02)
for ax, step in zip(axs.flatten(), selected):
    ax.imshow(frames[step], cmap=color_map)
    ax.set_title(f"Step {step}")
    ax.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.99])  # Reserve space for the title
plt.savefig("Kaleidoscope4", dpi=300, bbox_inches='tight')
plt.show()
