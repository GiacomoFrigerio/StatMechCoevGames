import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from numba import njit

# --- Simulation parameters ---
L = 200  # Lattice size
steps = 1000  # Total Monte Carlo steps
equilibration = 200  # Number of steps before measurement
K = 0.01  # Noise parameter
initial_coop_fraction = 0.5
b_values = np.linspace(1.0, 2.0, 41)  # Temptation to defect values

# --- Strategy and payoff settings ---
C, D = 1, 0
R, S, P = 1, 0, 0
kernel = np.ones((3, 3))  # 8 neighbors + self

def initialize_grid():
    return np.random.choice([C, D], size=(L, L), p=[initial_coop_fraction, 1 - initial_coop_fraction])

def compute_payoffs(grid, b):
    coop = (grid == C).astype(np.float32)
    defe = (grid == D).astype(np.float32)
    coop_sum = convolve(coop, kernel, mode='wrap')
    defe_sum = convolve(defe, kernel, mode='wrap')
    return coop * (R * coop_sum + S * defe_sum) + defe * (b * coop_sum + P * defe_sum)

@njit
def fermi_update_numba(grid, payoff, b, K):
    L = grid.shape[0]
    for _ in range(L * L):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        dx = np.random.choice(np.array([-1, 0, 1]))
        dy = np.random.choice(np.array([-1, 0, 1]))
        nx = (x + dx) % L
        ny = (y + dy) % L
        delta = payoff[nx, ny] - payoff[x, y]
        prob = 1.0 / (1.0 + np.exp(-delta / K))
        if np.random.rand() < prob:
            grid[x, y] = grid[nx, ny]
    return grid

# --- Run simulation ---
avg_coop_levels = []

for b in b_values:
    print("b-value: ", b)
    grid = initialize_grid().astype(np.uint8)
    coop_fractions = []

    for step in range(steps):
        payoff = compute_payoffs(grid, b)
        grid = fermi_update_numba(grid, payoff, b, K)
        if step >= equilibration:
            coop_fractions.append(np.mean(grid == C))

    avg_c = np.mean(coop_fractions)
    avg_coop_levels.append(avg_c)
    print(f"b = {b:.3f} -> c = {avg_c:.4f}")

# --- Plotting ---
plt.figure(figsize=(8, 5))
plt.plot(b_values, avg_coop_levels, 'ks-', label=f'MC (L={L}, K={K})')
plt.xlabel('Temptation to Defect (b)')
plt.ylabel('Density of Cooperators (c)')
plt.title('Density of Cooperators vs. b (Szabó–Tőke, Fermi Rule)')
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("Fig1_SzaboToke_Reproduction_Optimized.png", dpi=300)
plt.show()
