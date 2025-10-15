import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FFMpegWriter
from scipy.signal import convolve2d

# Game and strategy constants
R, S, P = 1, 0, 0
C, D = 1, 0
T = 1.81  # Temptation to defect

# Simulation settings
L = 1000
steps = 100
initial_coop_fraction = 0.9

# Color map for strategy transitions
color_map = colors.ListedColormap(['blue', 'red', 'yellow', 'green'])

def initialize_grid():
    return np.random.choice([C, D], size=(L, L), p=[initial_coop_fraction, 1 - initial_coop_fraction])

def compute_total_payoff(grid, T):
    coop = (grid == C).astype(int)
    defe = (grid == D).astype(int)
    kernel = np.ones((3, 3))

    coop_neighbors = convolve2d(coop, kernel, mode='same', boundary='fill', fillvalue=0)
    defe_neighbors = convolve2d(defe, kernel, mode='same', boundary='fill', fillvalue=0)

    coop_payoff = coop * (R * coop_neighbors + S * defe_neighbors)
    defe_payoff = defe * (T * coop_neighbors + P * defe_neighbors)

    return coop_payoff + defe_payoff

def update(grid, T):
    payoff = compute_total_payoff(grid, T)
    best_payoff = payoff.copy()
    best_strategy = grid.copy()

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted_grid = np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
            shifted_payoff = np.roll(np.roll(payoff, dx, axis=0), dy, axis=1)

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

def transition_colors(prev, curr):
    colors = np.zeros_like(curr)
    colors[(prev == C) & (curr == C)] = 0
    colors[(prev == D) & (curr == D)] = 1
    colors[(prev == C) & (curr == D)] = 2
    colors[(prev == D) & (curr == C)] = 3
    return colors

# Initialize simulation
grid = initialize_grid()
coop_fractions = []
frames = []

for _ in range(steps):
    coop_fractions.append(np.mean(grid == C))
    prev_grid = grid.copy()
    grid = update(grid, T)
    frames.append(transition_colors(prev_grid, grid))

# Animation setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
im = ax1.imshow(frames[0], cmap=color_map, animated=True)
ax1.set_title(f"Strategy Transitions (T = {T})")
ax1.axis('off')

line, = ax2.plot([], [], color='blue')
ax2.set_xlim(0, steps)
ax2.set_ylim(0, 1)
ax2.set_title("Cooperation Fraction Over Time")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Fraction")
ax2.grid(True)

def update_frame(i):
    im.set_array(frames[i])
    line.set_data(range(i + 1), coop_fractions[:i + 1])
    return im, line

writer = FFMpegWriter(fps=10, metadata=dict(artist='Evolutionary Game Simulation'), bitrate=1800)

with writer.saving(fig, "evolution_T1.8.mp4", dpi=200):
    for i in range(steps):
        update_frame(i)
        writer.grab_frame()

plt.close()
