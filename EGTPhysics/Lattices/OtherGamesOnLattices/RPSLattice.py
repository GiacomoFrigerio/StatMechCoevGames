import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Parameters ---
L = 100               # Lattice size
steps = 1000000         # Number of invasion events
record_every = 500    # How often to record strategy proportions

# Strategy labels
ROCK, PAPER, SCISSORS = 0, 1, 2
strategies = [ROCK, PAPER, SCISSORS]
colors = ListedColormap(['red', 'green', 'blue'])

# --- Set initial proportions (must sum to 1.0) ---
initial_proportions = [0.6, 0.2, 0.2]  # Rock, Paper, Scissors

# --- Initialize grid with given proportions ---
def initialize_grid():
    flat = np.random.choice(strategies, size=L*L, p=initial_proportions)
    return flat.reshape((L, L))

# --- Cyclic dominance rule ---
def beats(a, b):
    return (a - b) % 3 == 1

# --- Simulation ---
grid = initialize_grid()
time_series = []

for step in range(steps):
    x, y = np.random.randint(0, L, 2)
    direction = np.random.choice(['up', 'down', 'left', 'right'])
    if direction == 'up':
        nx, ny = (x - 1) % L, y
    elif direction == 'down':
        nx, ny = (x + 1) % L, y
    elif direction == 'left':
        nx, ny = x, (y - 1) % L
    else:
        nx, ny = x, (y + 1) % L

    if beats(grid[x, y], grid[nx, ny]):
        grid[nx, ny] = grid[x, y]

    if step % record_every == 0:
        unique, counts = np.unique(grid, return_counts=True)
        proportions = np.zeros(3)
        for val, count in zip(unique, counts):
            proportions[val] = count / (L * L)
        time_series.append(proportions)

# --- Prepare data ---
time_series = np.array(time_series)
timesteps = np.arange(0, steps, record_every)

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Final configuration
axs[0].imshow(grid, cmap=colors)
axs[0].set_title("Final Lattice Configuration")
axs[0].axis('off')

# Proportions over time
axs[1].plot(timesteps, time_series[:, ROCK], label="Rock", color='red')
axs[1].plot(timesteps, time_series[:, PAPER], label="Paper", color='green')
axs[1].plot(timesteps, time_series[:, SCISSORS], label="Scissors", color='blue')
axs[1].set_ylim(0, 1)
axs[1].set_title("Strategy Proportions Over Time")
axs[1].set_xlabel("Time step")
axs[1].set_ylabel("Fraction of population")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("RPSlattice", dpi = 100)
plt.show()
